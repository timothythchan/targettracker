"""
announcement_cars.py — Announcement Cumulative Abnormal Returns (CARs) for EarningsLens.

Computes cumulative abnormal returns around earnings call dates using the
standard market model. This module tests whether high-MT firms (those that
dropped more guidance targets) have significantly different announcement-window
returns compared to low-MT firms.

The paper's finding: announcement-window CARs are approximately zero for
high-MT firms, consistent with the market not immediately pricing the
obfuscation signal — the alpha accrues over subsequent months.

Market model
------------
For each firm i, estimate using a pre-event estimation window (e.g. t=-250 to t=-30):
    ret_{i,t} = alpha_i + beta_i * mkt_ret_t + epsilon_{i,t}

Then for each event date, compute abnormal returns:
    AR_{i,t} = ret_{i,t} - (alpha_hat_i + beta_hat_i * mkt_ret_t)

Cumulative abnormal return over window [tau_1, tau_2]:
    CAR_i = sum_{t=tau_1}^{tau_2} AR_{i,t}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False
    logger.warning("statsmodels not installed. Market model estimation unavailable.")

try:
    from scipy import stats as sp_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class AnnouncementCARs:
    """
    Compute and analyse cumulative abnormal returns around earnings call dates.

    Parameters
    ----------
    estimation_window:
        Tuple (start, end) in event-time days for the OLS estimation window.
        Convention: negative = days before event, -1 = one day before event.
        Default: (-250, -30) — a 221-day pre-event window.
    event_window:
        Tuple (start, end) in event-time days for the CAR accumulation window.
        Default: (-1, 1) — the canonical 3-day announcement window.
    market_col:
        Column name of the market return in the returns DataFrame.
    return_col:
        Column name of the individual stock return.

    Usage
    -----
    >>> acars = AnnouncementCARs(event_window=(-1, 1))
    >>> cars_df = acars.compute_cars(daily_returns, call_dates_df)
    >>> test_results = acars.test_announcement_effect(cars_df, mt_scores_df)
    """

    def __init__(
        self,
        estimation_window: Tuple[int, int] = (-250, -30),
        event_window: Tuple[int, int] = (-1, 1),
        market_col: str = "mkt_ret",
        return_col: str = "ret",
    ) -> None:
        self.est_start, self.est_end = estimation_window
        self.evt_start, self.evt_end = event_window
        self.market_col = market_col
        self.return_col = return_col

    # ------------------------------------------------------------------
    # CAR computation
    # ------------------------------------------------------------------

    def compute_cars(
        self,
        daily_returns: pd.DataFrame,
        call_dates: pd.DataFrame,
        firm_col: str = "company_id",
        date_col: str = "date",
        window: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute CARs for each firm-event observation in ``call_dates``.

        Parameters
        ----------
        daily_returns:
            Daily panel with columns: [date_col, firm_col, return_col, market_col].
            Should span the full estimation and event period.
        call_dates:
            DataFrame with columns [firm_col, date_col] giving each firm's
            earnings call date. May contain additional metadata columns.
        firm_col:
            Column name of the firm identifier.
        date_col:
            Column name of the date.
        window:
            Override event window for this call. Defaults to ``self.event_window``.

        Returns
        -------
        pd.DataFrame: One row per firm-event with columns:
            company_id, event_date, car, alpha_hat, beta_hat,
            n_est_obs, car_window_start, car_window_end
        """
        evt_start = window[0] if window else self.evt_start
        evt_end = window[1] if window else self.evt_end

        daily_returns = daily_returns.copy()
        daily_returns[date_col] = pd.to_datetime(daily_returns[date_col])
        daily_returns = daily_returns.sort_values([firm_col, date_col])

        records: List[Dict[str, Any]] = []

        for _, event_row in call_dates.iterrows():
            fid = event_row[firm_col]
            event_date = pd.to_datetime(event_row[date_col])

            # Get firm-specific daily returns
            firm_data = daily_returns[daily_returns[firm_col] == fid].copy()
            if firm_data.empty:
                logger.debug("No daily returns for firm %s", fid)
                continue

            # Align to event time
            firm_data = firm_data.sort_values(date_col).reset_index(drop=True)

            # Find event date index
            event_idx = self._find_event_index(firm_data[date_col].values, event_date)
            if event_idx is None:
                logger.debug(
                    "Could not locate event date %s for firm %s", event_date.date(), fid
                )
                continue

            # Extract estimation window data
            est_slice = self._extract_window(firm_data, event_idx, self.est_start, self.est_end)
            if est_slice is None or len(est_slice) < 30:
                logger.debug(
                    "Insufficient estimation data for %s @ %s (n=%d)",
                    fid,
                    event_date.date(),
                    len(est_slice) if est_slice is not None else 0,
                )
                continue

            # Estimate market model
            alpha_hat, beta_hat = self._estimate_market_model(est_slice)

            # Extract event window data
            evt_slice = self._extract_window(firm_data, event_idx, evt_start, evt_end)
            if evt_slice is None or evt_slice.empty:
                continue

            # Compute abnormal returns
            ar = evt_slice[self.return_col].values - (
                alpha_hat + beta_hat * evt_slice[self.market_col].values
            )
            car = float(ar.sum())

            records.append(
                {
                    firm_col: fid,
                    "event_date": event_date,
                    "car": car,
                    "alpha_hat": alpha_hat,
                    "beta_hat": beta_hat,
                    "n_est_obs": len(est_slice),
                    "car_window_start": evt_start,
                    "car_window_end": evt_end,
                    "n_event_days": len(evt_slice),
                }
            )

        cars_df = pd.DataFrame(records)
        logger.info(
            "compute_cars: computed %d CARs (window=[%d,%d])",
            len(cars_df),
            evt_start,
            evt_end,
        )
        return cars_df

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def test_announcement_effect(
        self,
        cars: pd.DataFrame,
        mt_scores: pd.DataFrame,
        mt_col: str = "mt_score_strict",
        firm_col: str = "company_id",
        date_col: str = "event_date",
        n_groups: int = 5,
    ) -> Dict[str, Any]:
        """
        Test whether high-MT firms have different CARs from low-MT firms.

        Tests:
            1. Cross-sectional regression: CAR = a + b * MT + controls + e
            2. Sort-based: mean CAR in MT quintile 5 vs. quintile 1
            3. t-test: difference in mean CARs (high-MT vs low-MT)

        Parameters
        ----------
        cars:
            Output of :meth:`compute_cars`.
        mt_scores:
            DataFrame with [firm_col, date_col (quarter or event_date), mt_col].
        mt_col:
            MT score column name in mt_scores.
        firm_col:
            Firm identifier column.
        date_col:
            Date or quarter column for merging.
        n_groups:
            Number of MT quantile groups for sort-based test.

        Returns
        -------
        Dict with keys:
            regression         : Dict — {alpha, beta_mt, t_mt, p_mt, n_obs}
            sort_test          : pd.DataFrame — mean CAR by MT quintile
            ttest_high_vs_low  : Dict — {mean_high, mean_low, t_stat, p_value}
            conclusion         : str — summary interpretation
        """
        # Merge CARs with MT scores
        mt = mt_scores[[firm_col, mt_col]].copy()

        # Try to merge on firm only (most lenient)
        merged = cars.merge(mt, on=firm_col, how="inner").dropna(subset=["car", mt_col])

        if merged.empty:
            logger.warning("test_announcement_effect: no merged observations")
            return {"regression": {}, "sort_test": pd.DataFrame(),
                    "ttest_high_vs_low": {}, "conclusion": "Insufficient data"}

        n = len(merged)
        logger.info("test_announcement_effect: n=%d merged observations", n)

        # 1. Cross-sectional regression: CAR ~ MT
        reg_result = self._regress_car_on_mt(merged, mt_col)

        # 2. Sort-based test
        merged["mt_group"] = pd.qcut(
            merged[mt_col], q=n_groups, labels=False, duplicates="drop"
        ) + 1
        sort_test = (
            merged.groupby("mt_group")["car"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_car", "std": "std_car", "count": "n"})
        )
        sort_test["t_vs_zero"] = sort_test["mean_car"] / (
            sort_test["std_car"] / np.sqrt(sort_test["n"])
        )

        # 3. t-test: high (top quintile) vs. low (bottom quintile)
        ttest_result = self._ttest_high_vs_low(merged, mt_col, n_groups)

        # Conclusion
        beta_mt = reg_result.get("beta_mt", np.nan)
        p_mt = reg_result.get("p_mt", np.nan)
        is_zero = (
            not np.isnan(p_mt) and p_mt > 0.10
        )  # fail to reject H0: beta = 0
        if is_zero:
            conclusion = (
                "Announcement-window CARs are NOT significantly related to MT scores "
                f"(beta_MT={beta_mt:.4f}, p={p_mt:.3f}), consistent with the paper's "
                "finding that the market does not immediately price MT obfuscation."
            )
        else:
            conclusion = (
                f"Announcement-window CARs ARE significantly related to MT scores "
                f"(beta_MT={beta_mt:.4f}, p={p_mt:.3f})."
            )

        logger.info("CARs test conclusion: %s", conclusion[:80])

        return {
            "regression": reg_result,
            "sort_test": sort_test,
            "ttest_high_vs_low": ttest_result,
            "conclusion": conclusion,
            "n_observations": n,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_market_model(
        self, data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Estimate alpha and beta from OLS regression of firm ret on market ret.

        Parameters
        ----------
        data: DataFrame with ``return_col`` and ``market_col`` columns.

        Returns
        -------
        Tuple[float, float]: (alpha_hat, beta_hat).
        """
        y = data[self.return_col].values
        x = data[self.market_col].values

        # Remove NaNs
        mask = ~(np.isnan(y) | np.isnan(x))
        y, x = y[mask], x[mask]

        if len(y) < 10:
            return 0.0, 1.0  # default: CAPM beta = 1

        if _SM_AVAILABLE:
            try:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                alpha_hat = float(model.params[0])
                beta_hat = float(model.params[1])
                return alpha_hat, beta_hat
            except Exception:
                pass

        # Fallback: manual OLS
        x_dm = x - x.mean()
        y_dm = y - y.mean()
        beta_hat = float(np.dot(x_dm, y_dm) / (np.dot(x_dm, x_dm) + 1e-12))
        alpha_hat = float(y.mean() - beta_hat * x.mean())
        return alpha_hat, beta_hat

    @staticmethod
    def _find_event_index(
        dates: np.ndarray, event_date: pd.Timestamp
    ) -> Optional[int]:
        """
        Return the index in a sorted date array closest to ``event_date``.

        Looks for an exact match first; then nearest within 3 calendar days.

        Parameters
        ----------
        dates:      Sorted numpy array of datetime64 values.
        event_date: Target event date.

        Returns
        -------
        Optional[int]: Integer index or None if no suitable date found.
        """
        dates_ts = pd.to_datetime(dates)
        if event_date in dates_ts:
            return int(np.where(dates_ts == event_date)[0][0])

        # Nearest within ±3 calendar days
        diffs = np.abs((dates_ts - event_date).days)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= 3:
            return idx
        return None

    def _extract_window(
        self,
        firm_data: pd.DataFrame,
        event_idx: int,
        start: int,
        end: int,
    ) -> Optional[pd.DataFrame]:
        """
        Extract a sub-window of rows relative to ``event_idx``.

        Parameters
        ----------
        firm_data:  Sorted firm daily DataFrame.
        event_idx:  Index of the event date row.
        start:      Start offset (negative = before event).
        end:        End offset (positive = after event).

        Returns
        -------
        Optional[pd.DataFrame]: Slice of firm_data or None if out of bounds.
        """
        idx_start = event_idx + start
        idx_end = event_idx + end + 1  # inclusive end

        if idx_start < 0:
            idx_start = 0
        if idx_end > len(firm_data):
            idx_end = len(firm_data)

        if idx_start >= idx_end:
            return None

        return firm_data.iloc[idx_start:idx_end]

    def _regress_car_on_mt(
        self, merged: pd.DataFrame, mt_col: str
    ) -> Dict[str, Any]:
        """Run OLS: CAR ~ constant + MT_score."""
        if not _SM_AVAILABLE:
            return {}

        try:
            y = merged["car"].values
            X = sm.add_constant(merged[mt_col].values)
            model = sm.OLS(y, X).fit()
            return {
                "alpha": float(model.params[0]),
                "beta_mt": float(model.params[1]),
                "t_alpha": float(model.tvalues[0]),
                "t_mt": float(model.tvalues[1]),
                "p_alpha": float(model.pvalues[0]),
                "p_mt": float(model.pvalues[1]),
                "r_squared": float(model.rsquared),
                "n_obs": int(model.nobs),
            }
        except Exception as exc:
            logger.warning("CAR regression failed: %s", exc)
            return {}

    @staticmethod
    def _ttest_high_vs_low(
        merged: pd.DataFrame,
        mt_col: str,
        n_groups: int,
    ) -> Dict[str, Any]:
        """t-test for difference in mean CARs: top vs. bottom MT quantile."""
        try:
            threshold_high = merged[mt_col].quantile(1 - 1 / n_groups)
            threshold_low = merged[mt_col].quantile(1 / n_groups)
            high_cars = merged.loc[merged[mt_col] >= threshold_high, "car"].dropna()
            low_cars = merged.loc[merged[mt_col] <= threshold_low, "car"].dropna()

            if len(high_cars) < 3 or len(low_cars) < 3:
                return {"mean_high": np.nan, "mean_low": np.nan,
                        "t_stat": np.nan, "p_value": np.nan}

            mean_high = float(high_cars.mean())
            mean_low = float(low_cars.mean())

            if _SCIPY_AVAILABLE:
                t_stat, p_value = sp_stats.ttest_ind(high_cars, low_cars, equal_var=False)
            else:
                # Manual Welch t-test
                n1, n2 = len(high_cars), len(low_cars)
                s1, s2 = high_cars.std(), low_cars.std()
                se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
                t_stat = (mean_high - mean_low) / se if se > 0 else np.nan
                p_value = np.nan

            return {
                "mean_high": mean_high,
                "mean_low": mean_low,
                "t_stat": float(t_stat),
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                "n_high": len(high_cars),
                "n_low": len(low_cars),
            }
        except Exception as exc:
            logger.warning("t-test failed: %s", exc)
            return {}
