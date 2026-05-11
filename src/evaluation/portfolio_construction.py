"""
portfolio_construction.py — Portfolio Construction for EarningsLens Financial Evaluation.

Constructs calendar-time quintile portfolios based on Moving Targets (MT) scores
and computes value-weighted returns for each portfolio. Supports both the spaCy
baseline MT and the LLM-enhanced semantic MT signal.

The paper's primary evaluation:
    - Each month, sort firms into quintile portfolios by their MT score.
    - Q5 = highest MT (most targets dropped — most suspicious)
    - Q1 = lowest MT (most targets maintained — most transparent)
    - Long-short spread: Q5 − Q1

This module produces the raw portfolio return series that are then passed to
``FamaMacBethRegression`` for risk-adjusted alpha estimation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of quintile portfolios
N_QUINTILES = 5


class PortfolioConstructor:
    """
    Construct calendar-time quintile portfolios from MT scores.

    Parameters
    ----------
    n_quintiles:
        Number of equal-sized portfolios. Defaults to 5 (quintiles).
    value_weighted:
        If True (default), compute value-weighted portfolio returns using
        lagged market capitalisation as weights. If False, compute equal-weighted.
    min_firms_per_portfolio:
        Minimum number of firms required in a portfolio for it to be included
        in the return series. Portfolios with fewer firms are set to NaN.

    Usage
    -----
    >>> pc = PortfolioConstructor()
    >>> assignments = pc.assign_quintiles(mt_df, score_col="mt_score_strict")
    >>> port_returns = pc.compute_calendar_time_returns(assignments, returns_df)
    >>> ls_returns = pc.compute_long_short_returns(port_returns)
    """

    def __init__(
        self,
        n_quintiles: int = N_QUINTILES,
        value_weighted: bool = True,
        min_firms_per_portfolio: int = 5,
    ) -> None:
        self.n_quintiles = n_quintiles
        self.value_weighted = value_weighted
        self.min_firms = min_firms_per_portfolio

    # ------------------------------------------------------------------
    # Quintile assignment
    # ------------------------------------------------------------------

    def assign_quintiles(
        self,
        mt_scores: pd.DataFrame,
        score_col: str = "mt_score_strict",
        date_col: str = "date",
        firm_col: str = "company_id",
    ) -> pd.DataFrame:
        """
        Assign firms to MT quintile portfolios each period.

        Quintile assignment is performed independently within each time period
        (month or quarter) to ensure the portfolios are investable at formation.

        Parameters
        ----------
        mt_scores:
            DataFrame with columns ``[date_col, firm_col, score_col, ...]``.
            Expects one row per firm-period observation.
        score_col:
            Column name of the MT score to use for ranking.
        date_col:
            Column name of the period identifier (date or quarter string).
        firm_col:
            Column name of the firm identifier.

        Returns
        -------
        pd.DataFrame: Input DataFrame augmented with:
            quintile     : int in [1, N] — 1 = lowest MT, N = highest MT
            quintile_label: str — "Q1" … "QN"
        """
        if score_col not in mt_scores.columns:
            raise ValueError(
                f"score_col '{score_col}' not found in mt_scores columns: "
                f"{mt_scores.columns.tolist()}"
            )

        df = mt_scores.copy()

        def _assign_quintile(group: pd.DataFrame) -> pd.DataFrame:
            valid_mask = group[score_col].notna()
            if valid_mask.sum() < self.n_quintiles:
                group = group.copy()
                group["quintile"] = np.nan
                group["quintile_label"] = np.nan
                return group

            # qcut with labels=False returns 0-indexed bins; add 1
            try:
                quintiles = pd.qcut(
                    group.loc[valid_mask, score_col],
                    q=self.n_quintiles,
                    labels=False,
                    duplicates="drop",
                )
                group = group.copy()
                group.loc[valid_mask, "quintile"] = quintiles + 1
                group.loc[~valid_mask, "quintile"] = np.nan
            except Exception as exc:
                logger.warning("qcut failed for period group: %s", exc)
                group = group.copy()
                group["quintile"] = np.nan

            group["quintile_label"] = group["quintile"].apply(
                lambda x: f"Q{int(x)}" if pd.notna(x) else np.nan
            )
            return group

        df = df.groupby(date_col, group_keys=False).apply(_assign_quintile)
        df["quintile"] = pd.to_numeric(df["quintile"], errors="coerce")

        logger.info(
            "assign_quintiles: %d observations across %d periods",
            len(df),
            df[date_col].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Calendar-time portfolio returns
    # ------------------------------------------------------------------

    def compute_calendar_time_returns(
        self,
        quintile_assignments: pd.DataFrame,
        returns_data: pd.DataFrame,
        date_col: str = "date",
        firm_col: str = "company_id",
        return_col: str = "ret",
        weight_col: str = "mktcap_lag",
        holding_period_months: int = 1,
    ) -> pd.DataFrame:
        """
        Compute value-weighted (or equal-weighted) monthly returns for each quintile.

        The function merges quintile assignments with forward returns, then for
        each (period, quintile) cell, computes the weighted average return.

        Parameters
        ----------
        quintile_assignments:
            Output of :meth:`assign_quintiles` — must contain ``date_col``,
            ``firm_col``, and ``quintile``.
        returns_data:
            DataFrame with ``[date_col, firm_col, return_col]`` and optionally
            ``weight_col`` for value-weighting.
        date_col:
            Period identifier column (must match across both DataFrames).
        firm_col:
            Firm identifier column.
        return_col:
            Column name of the holding-period stock return.
        weight_col:
            Column name of the value-weighting variable (e.g. lagged market cap).
            Only used when ``self.value_weighted=True``.
        holding_period_months:
            Not used directly (returns_data should already reflect the desired
            holding period). Reserved for future extension.

        Returns
        -------
        pd.DataFrame: Shape (n_periods, n_quintiles) with columns Q1 … QN
                      and index = period dates.
        """
        # Merge assignments with returns
        merged = quintile_assignments[[date_col, firm_col, "quintile"]].merge(
            returns_data[[date_col, firm_col, return_col]
                         + ([weight_col] if weight_col in returns_data.columns else [])],
            on=[date_col, firm_col],
            how="inner",
        )

        if merged.empty:
            logger.warning("compute_calendar_time_returns: merged DataFrame is empty")
            return pd.DataFrame()

        merged = merged.dropna(subset=["quintile", return_col])
        merged["quintile"] = merged["quintile"].astype(int)

        # Compute weighted returns
        records = []
        for period, period_df in merged.groupby(date_col):
            row: Dict[str, Any] = {"date": period}
            for q in range(1, self.n_quintiles + 1):
                qdf = period_df[period_df["quintile"] == q]
                if len(qdf) < self.min_firms:
                    row[f"Q{q}"] = np.nan
                elif self.value_weighted and weight_col in qdf.columns:
                    weights = qdf[weight_col].clip(lower=0)
                    total_w = weights.sum()
                    if total_w > 0:
                        row[f"Q{q}"] = float((qdf[return_col] * weights).sum() / total_w)
                    else:
                        row[f"Q{q}"] = float(qdf[return_col].mean())
                else:
                    row[f"Q{q}"] = float(qdf[return_col].mean())
            records.append(row)

        result = pd.DataFrame(records).set_index("date").sort_index()
        logger.info(
            "compute_calendar_time_returns: %d periods, %d quintile columns",
            len(result),
            len(result.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Long-short spread
    # ------------------------------------------------------------------

    def compute_long_short_returns(
        self,
        portfolio_returns: pd.DataFrame,
        long_col: Optional[str] = None,
        short_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute long-short (Q5 minus Q1) portfolio spread returns.

        The long-short portfolio is:
            Q5 (high MT = most targets dropped) MINUS Q1 (low MT = most transparent)

        A positive long-short return means firms dropping more targets *underperform*
        when the paper predicts they should be detected by the market.

        Parameters
        ----------
        portfolio_returns:
            Output of :meth:`compute_calendar_time_returns` with Q1 … QN columns.
        long_col:
            Column to go long. Defaults to "QN" (highest quintile).
        short_col:
            Column to go short. Defaults to "Q1" (lowest quintile).

        Returns
        -------
        pd.DataFrame: DataFrame with columns:
            Q_long   : float — long leg return
            Q_short  : float — short leg return
            LS       : float — long-short spread (long − short)
            LS_cum   : float — cumulative long-short return
        """
        if portfolio_returns.empty:
            return pd.DataFrame(columns=["Q_long", "Q_short", "LS", "LS_cum"])

        q_cols = sorted(
            [c for c in portfolio_returns.columns if c.startswith("Q")],
            key=lambda x: int(x[1:]),
        )
        if not q_cols:
            raise ValueError("portfolio_returns must have columns named Q1, Q2, …")

        long_col = long_col or q_cols[-1]   # QN
        short_col = short_col or q_cols[0]  # Q1

        if long_col not in portfolio_returns.columns:
            raise ValueError(f"long_col '{long_col}' not found")
        if short_col not in portfolio_returns.columns:
            raise ValueError(f"short_col '{short_col}' not found")

        ls_df = pd.DataFrame(index=portfolio_returns.index)
        ls_df["Q_long"] = portfolio_returns[long_col]
        ls_df["Q_short"] = portfolio_returns[short_col]
        ls_df["LS"] = ls_df["Q_long"] - ls_df["Q_short"]
        ls_df["LS_cum"] = (1 + ls_df["LS"].fillna(0)).cumprod() - 1

        logger.info(
            "compute_long_short_returns: long=%s, short=%s, "
            "mean_LS=%.4f%%, ann_sharpe=%.2f",
            long_col,
            short_col,
            float(ls_df["LS"].mean()) * 100,
            _annualised_sharpe(ls_df["LS"]),
        )
        return ls_df

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def portfolio_summary_stats(
        self, portfolio_returns: pd.DataFrame, annualise: bool = True
    ) -> pd.DataFrame:
        """
        Compute summary statistics for each quintile portfolio.

        Parameters
        ----------
        portfolio_returns:
            Output of :meth:`compute_calendar_time_returns`.
        annualise:
            If True, annualise mean returns and Sharpe ratio (×12 for monthly).

        Returns
        -------
        pd.DataFrame: Rows = Q1…QN + LS; Columns = mean_ret, std, sharpe, min, max.
        """
        q_cols = sorted(
            [c for c in portfolio_returns.columns if c.startswith("Q")],
            key=lambda x: int(x[1:]),
        )
        if not q_cols:
            return pd.DataFrame()

        ls_series = portfolio_returns[q_cols[-1]] - portfolio_returns[q_cols[0]]
        all_series = {q: portfolio_returns[q] for q in q_cols}
        all_series["LS"] = ls_series

        rows = []
        factor = 12 if annualise else 1
        for label, series in all_series.items():
            s = series.dropna()
            mean_r = float(s.mean()) * factor
            std_r = float(s.std()) * (factor ** 0.5)
            sharpe = mean_r / std_r if std_r > 0 else np.nan
            rows.append(
                {
                    "portfolio": label,
                    "mean_return": mean_r,
                    "std": std_r,
                    "sharpe": sharpe,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "n_obs": len(s),
                }
            )

        return pd.DataFrame(rows).set_index("portfolio")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _annualised_sharpe(monthly_returns: pd.Series) -> float:
    """Compute annualised Sharpe ratio from a monthly return series."""
    s = monthly_returns.dropna()
    if len(s) < 2 or s.std() == 0:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(12))
