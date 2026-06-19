"""
semantic_matcher.py — Semantic Continuity Matcher for EarningsLens RAG module.

Wraps TargetVectorStore and extends it with Moving Targets (MT) computation
using semantic similarity. This module replaces exact-string matching with
embedding-based continuity scoring, enabling detection of rephrased targets
that exact matching would falsely classify as dropped.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .vector_store import TargetVectorStore, DEFAULT_THRESHOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default threshold grid for calibration
# ---------------------------------------------------------------------------
_MAINTAINED_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]
_REPHRASED_GRID = [0.45, 0.50, 0.55, 0.60, 0.65]


class SemanticContinuityMatcher:
    """
    Semantic Continuity Matcher using sentence-transformer embeddings.

    This class provides:
    1. Semantic Moving Targets (MT) score — an embedding-based analogue
       of the paper's exact-match MT that is robust to target rephrasing.
    2. Threshold calibration via grid search against annotated validation data.

    The MT measure follows the original paper definition:
        MT = (number of targets dropped or rephrased) / (number of prior targets)

    Usage
    -----
    >>> matcher = SemanticContinuityMatcher()
    >>> result = matcher.compute_semantic_mt(
    ...     company_id="AAPL",
    ...     quarter="2023Q2",
    ...     current_targets=[{"metric_name": "Revenue growth", "context": "..."}],
    ...     threshold_config={"maintained": 0.80, "rephrased": 0.55},
    ... )
    >>> print(result["mt_score"])
    """

    def __init__(
        self,
        vector_store: Optional[TargetVectorStore] = None,
        persist_dir: Optional[str] = None,
        collection_name: str = "earnings_targets",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialise the SemanticContinuityMatcher.

        Parameters
        ----------
        vector_store:
            Pre-initialised TargetVectorStore instance. If None, a new one
            is constructed using the remaining keyword arguments.
        persist_dir:
            ChromaDB persist directory (only used when vector_store is None).
        collection_name:
            ChromaDB collection name (only used when vector_store is None).
        embedding_model:
            Sentence-transformer model identifier (only used when vector_store is None).
        default_thresholds:
            Dict with ``maintained`` and ``rephrased`` cosine threshold values.
            Defaults to ``DEFAULT_THRESHOLDS`` from vector_store module.
        """
        if vector_store is not None:
            self.store = vector_store
        else:
            kwargs: Dict[str, Any] = {"collection_name": collection_name, "embedding_model": embedding_model}
            if persist_dir is not None:
                kwargs["persist_dir"] = persist_dir
            self.store = TargetVectorStore(**kwargs)

        self.thresholds = default_thresholds or dict(DEFAULT_THRESHOLDS)

    # ------------------------------------------------------------------
    # Semantic MT computation
    # ------------------------------------------------------------------

    def compute_semantic_mt(
        self,
        company_id: str,
        quarter: str,
        current_targets: List[Dict[str, Any]],
        threshold_config: Optional[Dict[str, float]] = None,
        n_prior_quarters: int = 4,
        index_current: bool = True,
        historical_lag: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute the semantic Moving Targets (MT) measure for a single company-quarter.

        This is the core RAG-enhanced analogue of the baseline MT.  Historical
        targets are retrieved from ChromaDB and each is classified as
        *maintained*, *rephrased*, or *dropped* via cosine similarity with the
        current quarter's targets.

        Two retrieval modes:
        - **Window mode** (``historical_lag=None``, default): retrieve the
          union of targets from the prior ``n_prior_quarters`` quarters
          (t-1 .. t-n_prior_quarters). Backward-compatible legacy behavior.
        - **Paper-strict t-k mode** (``historical_lag`` set, e.g. 4): retrieve
          ONLY the single quarter at exact lag ``historical_lag`` (t-k).
          This matches the Moving Targets paper definition (k=4).

        MT is defined as the fraction of historical targets that are dropped
        (or dropped + rephrased for the lenient variant). We report both.

        Parameters
        ----------
        company_id:
            Unique company identifier (e.g. ticker or CIK).
        quarter:
            Current fiscal quarter string (e.g. "2023Q2").
        current_targets:
            List of current-quarter target dicts. Each dict must contain at
            minimum ``metric_name`` (str) and optionally ``context`` (str).
        threshold_config:
            Dict with ``maintained`` and ``rephrased`` cosine thresholds.
            Falls back to instance-level defaults.
        n_prior_quarters:
            Number of prior quarters to retrieve (window mode only).
            Ignored when ``historical_lag`` is provided.
        index_current:
            If True (default), index ``current_targets`` into the vector store
            so future quarters can retrieve them. Set to False when the caller
            has already pre-indexed the full corpus via
            ``TargetVectorStore.build_full_index`` to avoid double-indexing
            (which is much faster on large batches).
        historical_lag:
            If provided, switch to paper-strict t-k mode and retrieve targets
            from the single quarter at exact lag ``historical_lag`` (e.g. 4
            for the canonical t-4 setup). Default None (window mode).

        Returns
        -------
        Dict with keys:
            company_id          : str
            quarter             : str
            mt_score_strict     : float  — fraction of hist targets fully dropped
            mt_score_lenient    : float  — fraction dropped + rephrased
            n_current           : int    — number of current targets
            n_historical        : int    — number of historical targets examined
            maintained          : List[Dict]  — detailed per-target records
            rephrased           : List[Dict]
            dropped             : List[Dict]
            similarity_matrix   : pd.DataFrame  — full pairwise cosine matrix
        """
        thresholds = threshold_config or self.thresholds

        # Index current targets so they are available for future lookups.
        # Skip when caller has pre-indexed the corpus (build_full_index path).
        if index_current:
            self.store.index_targets(company_id, quarter, current_targets)

        # Retrieve historical targets.
        # When historical_lag is provided, use paper-strict single-quarter
        # retrieval at exact lag k. Otherwise fall back to the legacy window
        # of n_prior_quarters (union of t-1..t-n).
        if historical_lag is not None:
            historical = self.store.query_historical_targets(
                company_id,
                quarter,
                n_quarters=historical_lag,
                lag_only=True,
            )
        else:
            historical = self.store.query_historical_targets(
                company_id, quarter, n_quarters=n_prior_quarters
            )

        if not historical:
            logger.warning(
                "No historical targets found for %s prior to %s", company_id, quarter
            )
            return self._empty_result(company_id, quarter, current_targets)

        # Compute pairwise similarity matrix
        sim_matrix = self.store.compute_similarity(current_targets, historical)

        # Classify
        continuity = self.store.classify_continuity(sim_matrix, thresholds)

        n_historical = len(historical)
        n_dropped = len(continuity["dropped"])
        n_rephrased = len(continuity["rephrased"])

        mt_strict = n_dropped / n_historical if n_historical > 0 else 0.0
        mt_lenient = (n_dropped + n_rephrased) / n_historical if n_historical > 0 else 0.0

        # Build per-target detail records
        maintained_records = self._build_target_records(
            continuity["maintained"], continuity["details"], historical, "maintained"
        )
        rephrased_records = self._build_target_records(
            continuity["rephrased"], continuity["details"], historical, "rephrased"
        )
        dropped_records = self._build_target_records(
            continuity["dropped"], continuity["details"], historical, "dropped"
        )

        logger.info(
            "SemanticMT for %s %s: strict=%.3f, lenient=%.3f "
            "(hist=%d, maintained=%d, rephrased=%d, dropped=%d)",
            company_id,
            quarter,
            mt_strict,
            mt_lenient,
            n_historical,
            len(maintained_records),
            len(rephrased_records),
            len(dropped_records),
        )

        return {
            "company_id": company_id,
            "quarter": quarter,
            "mt_score_strict": mt_strict,
            "mt_score_lenient": mt_lenient,
            "n_current": len(current_targets),
            "n_historical": n_historical,
            "n_maintained": len(maintained_records),
            "n_rephrased": len(rephrased_records),
            "n_dropped": len(dropped_records),
            "historical_lag": historical_lag,
            "maintained": maintained_records,
            "rephrased": rephrased_records,
            "dropped": dropped_records,
            "similarity_matrix": sim_matrix,
        }

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def calibrate_thresholds(
        self,
        validation_data: List[Dict[str, Any]],
        maintained_grid: Optional[List[float]] = None,
        rephrased_grid: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Grid search over cosine threshold values to minimise false positive drop rate.

        The validation dataset must contain annotated pairs of consecutive quarters
        with manually labelled continuity.

        Parameters
        ----------
        validation_data:
            List of dicts, each with keys:
                company_id     : str
                quarter        : str  — the *current* quarter
                current_targets: List[Dict]
                historical_targets: List[Dict]  — aligned prior-quarter targets
                labels         : Dict[str, str]  — ``{hist_metric_name: "maintained"
                                                         | "rephrased" | "dropped"}``
        maintained_grid:
            Grid of ``maintained`` threshold values to search. Defaults to
            ``_MAINTAINED_GRID``.
        rephrased_grid:
            Grid of ``rephrased`` threshold values to search. Defaults to
            ``_REPHRASED_GRID``.

        Returns
        -------
        Dict with keys:
            best_maintained     : float
            best_rephrased      : float
            best_fp_rate        : float  — false positive drop rate at optimum
            calibration_results : pd.DataFrame  — full grid search results
        """
        m_grid = maintained_grid or _MAINTAINED_GRID
        r_grid = rephrased_grid or _REPHRASED_GRID

        records = []

        for m_thr, r_thr in itertools.product(m_grid, r_grid):
            if r_thr >= m_thr:
                # Thresholds must be ordered: rephrased < maintained
                continue

            thresholds = {"maintained": m_thr, "rephrased": r_thr}
            fp_drops = 0  # falsely classified as dropped
            total_not_dropped = 0  # ground truth: maintained or rephrased

            for sample in validation_data:
                current = sample["current_targets"]
                historical = sample["historical_targets"]
                labels: Dict[str, str] = sample.get("labels", {})

                sim_matrix = self.store.compute_similarity(current, historical)
                continuity = self.store.classify_continuity(sim_matrix, thresholds)

                predicted_dropped = set(continuity["dropped"])
                for hist_label, true_class in labels.items():
                    # Find matching column label (partial match on metric_name)
                    col_match = self._find_column(hist_label, sim_matrix.columns.tolist())
                    if col_match is None:
                        continue
                    if true_class in ("maintained", "rephrased"):
                        total_not_dropped += 1
                        if col_match in predicted_dropped:
                            fp_drops += 1

            fp_rate = fp_drops / total_not_dropped if total_not_dropped > 0 else 1.0
            records.append(
                {
                    "maintained_thr": m_thr,
                    "rephrased_thr": r_thr,
                    "fp_rate": fp_rate,
                    "fp_drops": fp_drops,
                    "total_not_dropped": total_not_dropped,
                }
            )

        if not records:
            logger.warning("Calibration produced no valid threshold combinations.")
            return {
                "best_maintained": self.thresholds["maintained"],
                "best_rephrased": self.thresholds["rephrased"],
                "best_fp_rate": float("nan"),
                "calibration_results": pd.DataFrame(),
            }

        results_df = pd.DataFrame(records).sort_values("fp_rate")
        best = results_df.iloc[0]

        # Update instance thresholds
        self.thresholds["maintained"] = float(best["maintained_thr"])
        self.thresholds["rephrased"] = float(best["rephrased_thr"])

        logger.info(
            "Calibration complete: best maintained=%.2f, rephrased=%.2f, FP rate=%.4f",
            self.thresholds["maintained"],
            self.thresholds["rephrased"],
            float(best["fp_rate"]),
        )

        return {
            "best_maintained": self.thresholds["maintained"],
            "best_rephrased": self.thresholds["rephrased"],
            "best_fp_rate": float(best["fp_rate"]),
            "calibration_results": results_df,
        }

    # ------------------------------------------------------------------
    # Batch semantic MT
    # ------------------------------------------------------------------

    def compute_batch_semantic_mt(
        self,
        company_quarters: List[Dict[str, Any]],
        threshold_config: Optional[Dict[str, float]] = None,
        n_prior_quarters: int = 4,
        index_current: bool = True,
        historical_lag: Optional[int] = None,
        return_per_pair: bool = False,
    ) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
        """
        Compute semantic MT scores for a list of company-quarter observations.

        This is the production batch driver used by ``scripts/run_rag_matching.py``
        (port of NB04 Cell 40). It is a thin wrapper around
        ``compute_semantic_mt`` that aggregates one row per company-quarter
        and, optionally, a long-form per-pair similarity DataFrame matching
        NB04's ``per_pair_sims.parquet`` output schema.

        Parameters
        ----------
        company_quarters:
            List of dicts with keys: ``company_id``, ``quarter``,
            ``current_targets``.
        threshold_config:
            Cosine threshold overrides.
        n_prior_quarters:
            Window-mode prior-quarter count. Ignored when ``historical_lag``
            is set.
        index_current:
            Pass-through to ``compute_semantic_mt``. Set False when the caller
            has already pre-indexed the corpus via ``TargetVectorStore.build_full_index``.
        historical_lag:
            Paper-strict t-k retrieval lag. None for window mode (legacy),
            4 for the canonical Moving Targets t-4 setup.
        return_per_pair:
            When True, also return the long-form per-pair similarity log
            (current_label, historical_label, similarity, company_id, quarter).
            This is what NB04 Cell 40 wrote to ``per_pair_sims.parquet``.

        Returns
        -------
        pd.DataFrame
            One row per company-quarter with MT scores and counts.
        Tuple[pd.DataFrame, pd.DataFrame]
            (mt_df, per_pair_df) when ``return_per_pair=True``.
        """
        rows: List[Dict[str, Any]] = []
        per_pair_rows: List[Dict[str, Any]] = []

        for cq in company_quarters:
            try:
                result = self.compute_semantic_mt(
                    company_id=cq["company_id"],
                    quarter=cq["quarter"],
                    current_targets=cq["current_targets"],
                    threshold_config=threshold_config,
                    n_prior_quarters=n_prior_quarters,
                    index_current=index_current,
                    historical_lag=historical_lag,
                )
                rows.append(
                    {
                        "company_id": result["company_id"],
                        "quarter": result["quarter"],
                        "mt_score_strict": result["mt_score_strict"],
                        "mt_score_lenient": result["mt_score_lenient"],
                        "n_current": result["n_current"],
                        "n_historical": result["n_historical"],
                        "n_maintained": len(result["maintained"]),
                        "n_rephrased": len(result["rephrased"]),
                        "n_dropped": len(result["dropped"]),
                    }
                )

                if return_per_pair:
                    sim_df = result.get("similarity_matrix")
                    if sim_df is not None and not sim_df.empty:
                        # current targets are columns, historical are rows
                        for hist_label in sim_df.index:
                            for cur_label in sim_df.columns:
                                per_pair_rows.append({
                                    "current_label": cur_label,
                                    "historical_label": hist_label,
                                    "similarity": float(sim_df.loc[hist_label, cur_label]),
                                    "company_id": result["company_id"],
                                    "quarter": result["quarter"],
                                })
            except Exception as exc:
                logger.error(
                    "Failed to compute semantic MT for %s %s: %s",
                    cq.get("company_id"),
                    cq.get("quarter"),
                    exc,
                )

        mt_df = pd.DataFrame(rows)
        if return_per_pair:
            return mt_df, pd.DataFrame(per_pair_rows)
        return mt_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        company_id: str,
        quarter: str,
        current_targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a zero-valued MT result structure."""
        return {
            "company_id": company_id,
            "quarter": quarter,
            "mt_score_strict": 0.0,
            "mt_score_lenient": 0.0,
            "n_current": len(current_targets),
            "n_historical": 0,
            "n_maintained": 0,
            "n_rephrased": 0,
            "n_dropped": 0,
            "historical_lag": None,
            "maintained": [],
            "rephrased": [],
            "dropped": [],
            "similarity_matrix": pd.DataFrame(),
        }

    @staticmethod
    def _build_target_records(
        target_labels: List[str],
        details: Dict[str, Dict[str, Any]],
        historical_targets: List[Dict[str, Any]],
        match_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Build enriched target records by merging detail info with original metadata.

        Parameters
        ----------
        target_labels:      Labels from the continuity classification result.
        details:            Per-target detail dict from classify_continuity.
        historical_targets: Full historical target dicts for metadata lookup.
        match_type:         Classification label string.

        Returns
        -------
        List[Dict]: Enriched target records.
        """
        # Build lookup from label to historical metadata
        hist_lookup: Dict[str, Dict[str, Any]] = {}
        for ht in historical_targets:
            label = (
                f"{ht.get('metric_name', '')} ({ht.get('fiscal_quarter', '')})"
            )
            hist_lookup[label] = ht

        records = []
        for lbl in target_labels:
            detail = details.get(lbl, {})
            hist_meta = hist_lookup.get(lbl, {})
            record = {
                "label": lbl,
                "match_type": match_type,
                "best_match_score": detail.get("best_match_score", 0.0),
                "best_match_target": detail.get("best_match_current", ""),
                "metric_name": hist_meta.get("metric_name", lbl.split(" (")[0]),
                "fiscal_quarter": hist_meta.get("fiscal_quarter", ""),
                "is_financial": hist_meta.get("is_financial", False),
                "target_type": hist_meta.get("target_type", "unknown"),
                "context": hist_meta.get("context", ""),
            }
            records.append(record)
        return records

    @staticmethod
    def _find_column(metric_name: str, columns: List[str]) -> Optional[str]:
        """
        Find the first column label that contains ``metric_name`` as a substring.

        Parameters
        ----------
        metric_name: Metric name to search for.
        columns:     Available column labels.

        Returns
        -------
        Optional[str]: Matching column or None.
        """
        for col in columns:
            if metric_name.lower() in col.lower():
                return col
        return None
