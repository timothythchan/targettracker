"""
comparison.py — Signal Comparison for EarningsLens Financial Evaluation.

Provides systematic comparison between the spaCy baseline extractor and the
LLM-based extractor across three dimensions:
    1. Extraction quality (precision, recall, F1 against ground truth)
    2. MT signal correlation (Pearson and Spearman rank correlation)
    3. Financial alpha comparison (Fama-MacBeth alpha and t-statistic)

The primary research question is: does the LLM extractor produce an MT signal
with meaningfully higher financial predictability than the spaCy baseline?
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import stats as sp_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Rank correlation will be unavailable.")


class SignalComparison:
    """
    Systematic comparison of spaCy baseline vs. LLM-enhanced MT signals.

    Usage
    -----
    >>> sc = SignalComparison()
    >>> quality = sc.compare_extraction_quality(spacy_tgts, llm_tgts, ground_truth)
    >>> correlation = sc.compare_mt_signals(spacy_mt_df, llm_mt_df)
    >>> alpha_comp = sc.compare_alpha(spacy_alpha_dict, llm_alpha_dict)
    >>> full = sc.generate_comparison_report()
    """

    def __init__(self) -> None:
        # Storage for incremental results — populated as methods are called
        self._quality_results: Optional[Dict[str, Any]] = None
        self._signal_correlation: Optional[Dict[str, Any]] = None
        self._alpha_comparison: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Extraction quality comparison
    # ------------------------------------------------------------------

    def compare_extraction_quality(
        self,
        spacy_targets: List[Dict[str, Any]],
        llm_targets: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        match_field: str = "metric_name",
        fuzzy_match: bool = True,
        fuzzy_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Compute precision, recall, and F1 for both extractors against ground truth.

        Parameters
        ----------
        spacy_targets:  List of target dicts from the spaCy extractor.
        llm_targets:    List of target dicts from the LLM extractor.
        ground_truth:   List of annotated ground-truth target dicts.
        match_field:    Field used for matching (default: "metric_name").
        fuzzy_match:    If True, use substring/token overlap matching.
                        If False, use exact string matching.
        fuzzy_threshold: Minimum token overlap ratio for a fuzzy match.

        Returns
        -------
        Dict with keys:
            spacy  : Dict — {precision, recall, f1, n_predicted, n_true, n_matched}
            llm    : Dict — {precision, recall, f1, n_predicted, n_true, n_matched}
            delta_f1: float — (llm_f1 - spacy_f1)
            delta_precision: float
            delta_recall: float
        """
        gt_labels = [t.get(match_field, "").lower().strip() for t in ground_truth]
        spacy_labels = [t.get(match_field, "").lower().strip() for t in spacy_targets]
        llm_labels = [t.get(match_field, "").lower().strip() for t in llm_targets]

        n_true = len(gt_labels)

        # Match functions
        match_fn = _fuzzy_match_count if fuzzy_match else _exact_match_count

        spacy_matched = match_fn(spacy_labels, gt_labels, fuzzy_threshold)
        llm_matched = match_fn(llm_labels, gt_labels, fuzzy_threshold)

        spacy_metrics = _prf(
            n_predicted=len(spacy_labels),
            n_true=n_true,
            n_matched=spacy_matched,
        )
        llm_metrics = _prf(
            n_predicted=len(llm_labels),
            n_true=n_true,
            n_matched=llm_matched,
        )

        result: Dict[str, Any] = {
            "spacy": {**spacy_metrics, "n_predicted": len(spacy_labels), "n_matched": spacy_matched},
            "llm": {**llm_metrics, "n_predicted": len(llm_labels), "n_matched": llm_matched},
            "n_true": n_true,
            "delta_f1": llm_metrics["f1"] - spacy_metrics["f1"],
            "delta_precision": llm_metrics["precision"] - spacy_metrics["precision"],
            "delta_recall": llm_metrics["recall"] - spacy_metrics["recall"],
        }

        logger.info(
            "Extraction quality — spaCy F1=%.3f, LLM F1=%.3f, ΔF1=%.3f",
            spacy_metrics["f1"],
            llm_metrics["f1"],
            result["delta_f1"],
        )

        self._quality_results = result
        return result

    # ------------------------------------------------------------------
    # Signal correlation
    # ------------------------------------------------------------------

    def compare_mt_signals(
        self,
        spacy_mt: pd.DataFrame,
        llm_mt: pd.DataFrame,
        mt_col_spacy: str = "mt_score_strict",
        mt_col_llm: str = "mt_score_strict",
        id_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Pearson and Spearman rank correlations between MT signals.

        Parameters
        ----------
        spacy_mt:       DataFrame with spaCy MT scores.
        llm_mt:         DataFrame with LLM MT scores.
        mt_col_spacy:   MT score column name in spacy_mt.
        mt_col_llm:     MT score column name in llm_mt.
        id_cols:        Columns to use as merge keys. Defaults to
                        ["company_id", "date"] or ["company_id", "fiscal_quarter"].

        Returns
        -------
        Dict with keys:
            pearson_r         : float
            pearson_p         : float
            spearman_r        : float
            spearman_p        : float
            n_observations    : int
            mean_spacy_mt     : float
            mean_llm_mt       : float
            std_spacy_mt      : float
            std_llm_mt        : float
        """
        if id_cols is None:
            # Try to infer merge key columns
            common_candidates = ["company_id", "date", "fiscal_quarter", "quarter"]
            id_cols = [c for c in common_candidates
                       if c in spacy_mt.columns and c in llm_mt.columns]
            if not id_cols:
                id_cols = list(set(spacy_mt.columns) & set(llm_mt.columns) - {mt_col_spacy})

        spacy_renamed = spacy_mt[id_cols + [mt_col_spacy]].rename(
            columns={mt_col_spacy: "mt_spacy"}
        )
        llm_renamed = llm_mt[id_cols + [mt_col_llm]].rename(
            columns={mt_col_llm: "mt_llm"}
        )

        merged = spacy_renamed.merge(llm_renamed, on=id_cols, how="inner")
        merged = merged.dropna(subset=["mt_spacy", "mt_llm"])
        n = len(merged)

        if n < 3:
            logger.warning("compare_mt_signals: too few observations (%d) for correlation", n)
            result = {
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "n_observations": n,
            }
            self._signal_correlation = result
            return result

        # Pearson correlation
        pearson_r, pearson_p = np.nan, np.nan
        if _SCIPY_AVAILABLE:
            pearson_r, pearson_p = sp_stats.pearsonr(merged["mt_spacy"], merged["mt_llm"])
        else:
            pearson_r = float(np.corrcoef(merged["mt_spacy"], merged["mt_llm"])[0, 1])

        # Spearman rank correlation
        spearman_r, spearman_p = np.nan, np.nan
        if _SCIPY_AVAILABLE:
            spearman_r, spearman_p = sp_stats.spearmanr(merged["mt_spacy"], merged["mt_llm"])

        result = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else np.nan,
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else np.nan,
            "n_observations": n,
            "mean_spacy_mt": float(merged["mt_spacy"].mean()),
            "mean_llm_mt": float(merged["mt_llm"].mean()),
            "std_spacy_mt": float(merged["mt_spacy"].std()),
            "std_llm_mt": float(merged["mt_llm"].std()),
        }

        logger.info(
            "MT signal correlation — Pearson r=%.3f, Spearman r=%.3f (n=%d)",
            result["pearson_r"],
            result["spearman_r"],
            n,
        )

        self._signal_correlation = result
        return result

    # ------------------------------------------------------------------
    # Alpha comparison
    # ------------------------------------------------------------------

    def compare_alpha(
        self,
        spacy_alpha: Dict[str, Any],
        llm_alpha: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Tabulate and compare Fama-MacBeth alphas from both extractors.

        Parameters
        ----------
        spacy_alpha:
            Dict from FamaMacBethRegression.run_full_analysis() for spaCy MT.
            Expected keys: mt_alpha, mt_tstat, mt_p_value, n_periods, n_firms_avg.
        llm_alpha:
            Dict from FamaMacBethRegression.run_full_analysis() for LLM MT.

        Returns
        -------
        Dict with keys:
            spacy_alpha       : float
            spacy_tstat       : float
            spacy_p_value     : float
            llm_alpha         : float
            llm_tstat         : float
            llm_p_value       : float
            alpha_improvement : float — llm_alpha - spacy_alpha
            tstat_improvement : float — llm_tstat - spacy_tstat
            llm_significant   : bool  — p < 0.05
            spacy_significant : bool  — p < 0.05
        """
        spacy_a = float(spacy_alpha.get("mt_alpha", np.nan))
        spacy_t = float(spacy_alpha.get("mt_tstat", np.nan))
        spacy_p = float(spacy_alpha.get("mt_p_value", np.nan))

        llm_a = float(llm_alpha.get("mt_alpha", np.nan))
        llm_t = float(llm_alpha.get("mt_tstat", np.nan))
        llm_p = float(llm_alpha.get("mt_p_value", np.nan))

        result: Dict[str, Any] = {
            "spacy_alpha": spacy_a,
            "spacy_tstat": spacy_t,
            "spacy_p_value": spacy_p,
            "llm_alpha": llm_a,
            "llm_tstat": llm_t,
            "llm_p_value": llm_p,
            "alpha_improvement": llm_a - spacy_a if not (np.isnan(llm_a) or np.isnan(spacy_a)) else np.nan,
            "tstat_improvement": llm_t - spacy_t if not (np.isnan(llm_t) or np.isnan(spacy_t)) else np.nan,
            "llm_significant": llm_p < 0.05 if not np.isnan(llm_p) else False,
            "spacy_significant": spacy_p < 0.05 if not np.isnan(spacy_p) else False,
        }

        logger.info(
            "Alpha comparison — spaCy: %.5f (t=%.2f), LLM: %.5f (t=%.2f)",
            spacy_a,
            spacy_t,
            llm_a,
            llm_t,
        )

        self._alpha_comparison = result
        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate a full comparison summary combining all three dimensions.

        Must call :meth:`compare_extraction_quality`, :meth:`compare_mt_signals`,
        and :meth:`compare_alpha` before calling this method.

        Returns
        -------
        Dict with keys:
            extraction_quality  : Dict (from compare_extraction_quality)
            signal_correlation  : Dict (from compare_mt_signals)
            alpha_comparison    : Dict (from compare_alpha)
            verdict             : str — "LLM_BETTER" | "SPACY_BETTER" | "SIMILAR"
            summary             : str — human-readable narrative
        """
        report: Dict[str, Any] = {
            "extraction_quality": self._quality_results,
            "signal_correlation": self._signal_correlation,
            "alpha_comparison": self._alpha_comparison,
        }

        # Determine verdict
        verdict = "SIMILAR"
        summary_parts: List[str] = []

        if self._quality_results is not None:
            df1 = self._quality_results.get("delta_f1", 0.0)
            summary_parts.append(
                f"LLM extractor achieves ΔF1={df1:+.3f} relative to spaCy baseline."
            )
            if df1 > 0.05:
                verdict = "LLM_BETTER"
            elif df1 < -0.05:
                verdict = "SPACY_BETTER"

        if self._signal_correlation is not None:
            r = self._signal_correlation.get("pearson_r", np.nan)
            summary_parts.append(
                f"MT signals are {'strongly' if abs(r) > 0.7 else 'moderately'} "
                f"correlated (Pearson r={r:.3f})."
            )

        if self._alpha_comparison is not None:
            llm_a = self._alpha_comparison.get("llm_alpha", np.nan)
            spacy_a = self._alpha_comparison.get("spacy_alpha", np.nan)
            llm_sig = self._alpha_comparison.get("llm_significant", False)
            spacy_sig = self._alpha_comparison.get("spacy_significant", False)
            summary_parts.append(
                f"LLM MT alpha={llm_a:.5f} ({'significant' if llm_sig else 'not significant'}); "
                f"spaCy MT alpha={spacy_a:.5f} ({'significant' if spacy_sig else 'not significant'})."
            )
            if not np.isnan(llm_a) and not np.isnan(spacy_a):
                if llm_a > spacy_a and llm_sig and not spacy_sig:
                    verdict = "LLM_BETTER"
                elif spacy_a > llm_a and spacy_sig and not llm_sig:
                    verdict = "SPACY_BETTER"

        report["verdict"] = verdict
        report["summary"] = " ".join(summary_parts)

        logger.info("Comparison report generated. Verdict: %s", verdict)
        return report

    # ------------------------------------------------------------------
    # Convenience: formatted table
    # ------------------------------------------------------------------

    def to_latex_table(self) -> str:
        """
        Produce a LaTeX-formatted comparison table for the paper appendix.

        Returns
        -------
        str: LaTeX tabular environment string.
        """
        lines = [
            r"\begin{tabular}{lrrrrr}",
            r"\hline",
            r"Metric & spaCy Precision & spaCy Recall & spaCy F1 & LLM Precision & LLM Recall & LLM F1 \\",
            r"\hline",
        ]

        if self._quality_results:
            s = self._quality_results.get("spacy", {})
            l = self._quality_results.get("llm", {})
            lines.append(
                f"Extraction & {s.get('precision', 0):.3f} & {s.get('recall', 0):.3f} "
                f"& {s.get('f1', 0):.3f} & {l.get('precision', 0):.3f} "
                f"& {l.get('recall', 0):.3f} & {l.get('f1', 0):.3f} \\\\"
            )

        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _exact_match_count(predicted: List[str], ground_truth: List[str], _: float) -> int:
    """Count exact string matches between predicted and ground truth lists."""
    gt_set = set(ground_truth)
    return sum(1 for p in predicted if p in gt_set)


def _fuzzy_match_count(
    predicted: List[str],
    ground_truth: List[str],
    threshold: float = 0.7,
) -> int:
    """
    Count fuzzy (token-overlap) matches using Jaccard similarity on word tokens.

    Each predicted label is matched against the best ground truth label.
    Matches are exclusive: each GT label can only be matched once.

    Parameters
    ----------
    predicted:  Predicted label strings.
    ground_truth: Ground truth label strings.
    threshold:  Minimum Jaccard similarity for a match.

    Returns
    -------
    int: Number of matched predictions.
    """
    def _tokenise(s: str) -> set:
        return set(s.lower().split())

    gt_tokens = [_tokenise(g) for g in ground_truth]
    matched_gt = set()
    count = 0

    for pred in predicted:
        pred_toks = _tokenise(pred)
        best_score = 0.0
        best_idx = -1
        for i, gt_toks in enumerate(gt_tokens):
            if i in matched_gt:
                continue
            union = pred_toks | gt_toks
            inter = pred_toks & gt_toks
            if not union:
                continue
            jac = len(inter) / len(union)
            if jac > best_score:
                best_score = jac
                best_idx = i
        if best_score >= threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            count += 1

    return count


def _prf(n_predicted: int, n_true: int, n_matched: int) -> Dict[str, float]:
    """Compute precision, recall, and F1 from match counts."""
    precision = n_matched / n_predicted if n_predicted > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}
