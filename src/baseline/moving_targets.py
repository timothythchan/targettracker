"""
MovingTargetsComputer — Layer 1 Baseline MT Measure Computation
===============================================================
Implements the Cohen & Nguyen (2024) "Moving Targets" (MT) measure using
exact string matching on normalized target text (the spaCy baseline approach).

MT_{i,t} = |T_{i,t-4} ∖ T_{i,t}| / |T_{i,t-4}|

where T_{i,t-4} ∖ T_{i,t} is the set of targets present four quarters ago
but NOT in the current quarter.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger("earningslens.baseline.mt")

# ---------------------------------------------------------------------------
# Quarter arithmetic helpers
# ---------------------------------------------------------------------------


def _quarter_to_int(fiscal_year: int, fiscal_quarter: int) -> int:
    """Encode (year, quarter) as an integer for arithmetic comparisons."""
    return fiscal_year * 4 + (fiscal_quarter - 1)


def _int_to_quarter(q_int: int) -> Tuple[int, int]:
    """Decode an integer back to (fiscal_year, fiscal_quarter)."""
    fiscal_quarter = (q_int % 4) + 1
    fiscal_year = q_int // 4
    return fiscal_year, fiscal_quarter


def _quarter_key(fiscal_year: int, fiscal_quarter: int) -> str:
    """Canonical string key: e.g. '2023Q3'."""
    return f"{fiscal_year}Q{fiscal_quarter}"


def _parse_quarter_key(key: str) -> Tuple[int, int]:
    """Parse '2023Q3' → (2023, 3)."""
    year_str, q_str = key.split("Q")
    return int(year_str), int(q_str)


# ---------------------------------------------------------------------------
# MovingTargetsComputer
# ---------------------------------------------------------------------------


class MovingTargetsComputer:
    """
    Compute the Moving Targets (MT) measure and related statistics for each
    (company, quarter) pair.

    The MT measure follows Cohen & Nguyen (2024):
      MT_{i,t} = |T_{i,t-4} ∖ T_{i,t}| / |T_{i,t-4}|

    Parameters
    ----------
    persistence_window : int
        Number of prior consecutive quarters used for the persistence flag
        (default: 12, as specified in the paper).
    """

    def __init__(self, persistence_window: int = 12) -> None:
        self.persistence_window = persistence_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_mt(
        self,
        target_sets: Dict[Tuple[str, str], List[Dict]],
    ) -> pd.DataFrame:
        """
        Compute MT measures for every (company_id, quarter) with sufficient
        history.

        Parameters
        ----------
        target_sets : dict
            Keys are ``(company_id, quarter_key)`` where ``quarter_key`` is
            e.g. ``'2023Q3'``.
            Values are lists of target dicts as returned by
            :class:`SpacyTargetExtractor`.

        Returns
        -------
        pd.DataFrame
            One row per (company_id, quarter) that has a valid t-4 observation.
            Columns:

            ==================  ==============================================
            company_id          str
            quarter             str  e.g. '2023Q3'
            fiscal_year         int
            fiscal_quarter      int
            n_targets           int  # targets at time t
            n_targets_tm4       int  # targets at time t-4
            n_dropped           int  # |T_{t-4} \\ T_t|
            n_new               int  # |T_t \\ T_{t-4}|
            mt_score            float  MT_{i,t} (NaN if n_targets_tm4 == 0)
            dropped_targets     list  normalized texts of dropped targets
            financial_drop_ratio  float  fraction of drops that are financial
            nonfinancial_drop_ratio float
            ==================  ==============================================
        """
        # Group keys by company
        companies: Dict[str, List[Tuple[int, int]]] = {}
        for (company_id, quarter_key) in target_sets:
            fy, fq = _parse_quarter_key(quarter_key)
            companies.setdefault(company_id, []).append((fy, fq))

        rows: List[Dict] = []

        for company_id, quarters in companies.items():
            # Sort quarters chronologically
            sorted_quarters = sorted(quarters, key=lambda x: _quarter_to_int(*x))

            for fy, fq in sorted_quarters:
                qkey = _quarter_key(fy, fq)
                tm4_int = _quarter_to_int(fy, fq) - 4
                tm4_fy, tm4_fq = _int_to_quarter(tm4_int)
                tm4_key = _quarter_key(tm4_fy, tm4_fq)

                current_targets = target_sets.get((company_id, qkey), [])
                past_targets = target_sets.get((company_id, tm4_key))

                if past_targets is None:
                    # No t-4 data — skip (can still record n_targets)
                    continue

                current_set = self._normalize_set(current_targets)
                past_set = self._normalize_set(past_targets)

                dropped_normalized = past_set - current_set
                new_normalized = current_set - past_set

                n_targets = len(current_set)
                n_targets_tm4 = len(past_set)
                n_dropped = len(dropped_normalized)
                n_new = len(new_normalized)

                mt_score = (
                    n_dropped / n_targets_tm4
                    if n_targets_tm4 > 0
                    else float("nan")
                )

                # Financial / non-financial breakdown of dropped targets
                fin_ratio, nonfin_ratio = self._compute_drop_ratios(
                    dropped_normalized, past_targets
                )

                rows.append(
                    {
                        "company_id": company_id,
                        "quarter": qkey,
                        "fiscal_year": fy,
                        "fiscal_quarter": fq,
                        "n_targets": n_targets,
                        "n_targets_tm4": n_targets_tm4,
                        "n_dropped": n_dropped,
                        "n_new": n_new,
                        "mt_score": mt_score,
                        "dropped_targets": sorted(dropped_normalized),
                        "financial_drop_ratio": fin_ratio,
                        "nonfinancial_drop_ratio": nonfin_ratio,
                    }
                )

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning("compute_mt: no rows produced — insufficient t-4 history.")
            return df

        df = df.sort_values(["company_id", "fiscal_year", "fiscal_quarter"]).reset_index(
            drop=True
        )

        logger.info(
            "compute_mt: %d (company, quarter) rows produced for %d companies.",
            len(df),
            df["company_id"].nunique(),
        )
        return df

    def compute_persistence(
        self,
        company_targets: Dict[str, List[Dict]],
        current_quarter: str,
    ) -> Dict[str, bool]:
        """
        Check whether each dropped target was present in ALL of the prior
        ``persistence_window`` consecutive quarters.

        Parameters
        ----------
        company_targets : dict
            Keys are quarter_keys (e.g. ``'2023Q3'``); values are target lists.
        current_quarter : str
            The reference quarter (e.g. ``'2023Q3'``).  Persistence is checked
            for the ``persistence_window`` quarters *before* this quarter.

        Returns
        -------
        dict
            Mapping from normalized target text → bool (True = persistent).
        """
        fy, fq = _parse_quarter_key(current_quarter)
        ref_int = _quarter_to_int(fy, fq)

        # Collect the set of normalized targets for each lookback quarter
        lookback_sets: List[Set[str]] = []
        for lag in range(1, self.persistence_window + 1):
            lag_fy, lag_fq = _int_to_quarter(ref_int - lag)
            lag_key = _quarter_key(lag_fy, lag_fq)
            lag_targets = company_targets.get(lag_key, [])
            lookback_sets.append(self._normalize_set(lag_targets))

        # A target is persistent iff it appears in ALL lookback quarters
        # that have data (require full window for True)
        if not lookback_sets:
            return {}

        # Gather all unique targets ever seen in the lookback window
        all_seen: Set[str] = set().union(*lookback_sets)

        persistence: Dict[str, bool] = {}
        for target_norm in all_seen:
            present_in_all = all(
                target_norm in q_set for q_set in lookback_sets
            )
            persistence[target_norm] = present_in_all

        return persistence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_set(self, targets: List[Dict]) -> Set[str]:
        """
        Return the set of normalized target texts for a list of target dicts.

        The ``target_text`` field is assumed to already be normalized
        (i.e. passed through ``SpacyTargetExtractor.normalize_target``
        before being stored).  If a ``normalized_text`` key is present it
        takes precedence; otherwise ``target_text`` is used as-is.
        """
        result: Set[str] = set()
        for t in targets:
            norm = t.get("normalized_text") or t.get("target_text", "")
            if norm:
                result.add(norm.strip().lower())
        return result

    def _compute_drop_ratios(
        self,
        dropped_normalized: Set[str],
        past_targets: List[Dict],
    ) -> Tuple[float, float]:
        """
        Compute financial and non-financial drop ratios.

        Parameters
        ----------
        dropped_normalized : set
            Normalized texts of dropped targets.
        past_targets : list
            Full target dicts from time t-4 (used to look up entity labels).

        Returns
        -------
        Tuple[float, float]
            ``(financial_drop_ratio, nonfinancial_drop_ratio)``
        """
        if not dropped_normalized:
            return (float("nan"), float("nan"))

        # Build lookup: normalized_text → is_financial
        is_fin_map: Dict[str, bool] = {}
        for t in past_targets:
            norm = t.get("normalized_text") or t.get("target_text", "")
            if norm:
                is_fin_map[norm.strip().lower()] = t.get("is_financial", False)

        n_financial = sum(
            1 for dn in dropped_normalized if is_fin_map.get(dn, False)
        )
        n_nonfin = len(dropped_normalized) - n_financial
        total = len(dropped_normalized)

        return (n_financial / total, n_nonfin / total)


# ---------------------------------------------------------------------------
# Convenience: augment target dicts with persistence flag
# ---------------------------------------------------------------------------


def add_persistence_flags(
    mt_df: pd.DataFrame,
    target_sets: Dict[Tuple[str, str], List[Dict]],
    persistence_window: int = 12,
) -> pd.DataFrame:
    """
    Augment an MT DataFrame (from :meth:`MovingTargetsComputer.compute_mt`)
    with a ``persistent_flag`` column.

    The column is a dict {target_norm: bool} for each row, indicating
    whether each dropped target was present in all prior ``persistence_window``
    quarters.

    Parameters
    ----------
    mt_df : pd.DataFrame
        Output of :meth:`MovingTargetsComputer.compute_mt`.
    target_sets : dict
        The same ``{(company_id, quarter_key): [targets]}`` dict used to
        produce ``mt_df``.
    persistence_window : int
        Lookback window (default 12 quarters).

    Returns
    -------
    pd.DataFrame
        ``mt_df`` with an additional ``persistent_flag`` column.
    """
    computer = MovingTargetsComputer(persistence_window=persistence_window)

    def _row_persistence(row: pd.Series) -> Dict[str, bool]:
        company_id = row["company_id"]
        quarter = row["quarter"]
        # Build company-specific target dict keyed by quarter
        company_targets = {
            qk: tgts
            for (cid, qk), tgts in target_sets.items()
            if cid == company_id
        }
        return computer.compute_persistence(company_targets, quarter)

    mt_df = mt_df.copy()
    mt_df["persistent_flag"] = mt_df.apply(_row_persistence, axis=1)
    return mt_df


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    # Minimal synthetic target sets for two companies over several quarters
    synthetic: Dict[Tuple[str, str], List[Dict]] = {
        ("AAPL", "2022Q1"): [
            {"target_text": "15 percent revenue growth", "is_financial": True},
            {"target_text": "2 billion free cash flow", "is_financial": True},
            {"target_text": "new iphone product launch", "is_financial": False},
        ],
        ("AAPL", "2023Q1"): [
            {"target_text": "15 percent revenue growth", "is_financial": True},
            # Dropped: "2 billion free cash flow" and "new iphone product launch"
            {"target_text": "40 percent gross margin", "is_financial": True},
        ],
        ("MSFT", "2022Q2"): [
            {"target_text": "20 percent ebitda margin", "is_financial": True},
            {"target_text": "cloud revenue target", "is_financial": False},
        ],
        ("MSFT", "2023Q2"): [
            {"target_text": "20 percent ebitda margin", "is_financial": True},
            {"target_text": "cloud revenue target", "is_financial": False},
            {"target_text": "ai integration product", "is_financial": False},
        ],
    }

    computer = MovingTargetsComputer()
    df = computer.compute_mt(synthetic)
    print("\nMT DataFrame:\n")
    print(df.to_string())

    # Persistence
    aapl_targets = {
        qk: tgts for (cid, qk), tgts in synthetic.items() if cid == "AAPL"
    }
    persistence = computer.compute_persistence(aapl_targets, "2023Q1")
    print("\nAAPL persistence @ 2023Q1:")
    for tgt, flag in persistence.items():
        print(f"  {tgt!r}: persistent={flag}")
