"""
fama_macbeth.py — Fama-MacBeth Cross-Sectional Regression for EarningsLens.

Implements the two-pass Fama-MacBeth (1973) regression procedure to estimate
the risk-adjusted alpha of the Moving Targets (MT) signal:

    Pass 1: Run cross-sectional regression each period t:
        ret_{i,t+1} = a_t + b_t * MT_{i,t} + c_t * Controls_{i,t} + e_{i,t}

    Pass 2: Time-series average the coefficients (b_1, b_2, ..., b_T)
            and compute Newey-West t-statistics with 6 lags.

Control variables (paper specification):
    - Size       : log(market capitalisation)
    - logBM      : log(book-to-market ratio)
    - Ret(-1,0)  : prior month return (momentum reversal)
    - Ret(-12,-2): 11-month cumulative return excluding last month (momentum)
    - SUE        : Standardised Unexpected Earnings

All continuous controls are winsorised at the 1st/99th percentile each period.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional statsmodels — hard required for FM regression
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False
    logger.warning("statsmodels not installed. Fama-MacBeth regression unavailable.")

# Winsorisation percentiles
_WIN_LOW = 0.01
_WIN_HIGH = 0.99

# Newey-West lag order (6 months as in typical FM papers)
_NW_LAGS = 6


class FamaMacBethRegression:
    """
    Fama-MacBeth (1973) two-pass cross-sectional regression estimator.

    Designed for monthly panel data. The key independent variable is the
    Moving Targets (MT) signal; control variables follow the paper specification.

    Usage
    -----
    >>> fm = FamaMacBethRegression()
    >>> panel = fm.prepare_panel(mt_scores, returns, fundamentals, ibes, factors)
    >>> results = fm.run_full_analysis(panel)
    >>> print(results["mt_alpha"], results["mt_tstat"])
    """

    def __init__(
        self,
        winsorise: bool = True,
        nw_lags: int = _NW_LAGS,
        add_constant: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        winsorise:    Apply cross-sectional winsorisation to continuous vars.
        nw_lags:      Number of lags for Newey-West standard error correction.
        add_constant: Add intercept to each cross-sectional regression.
        """
        if not _SM_AVAILABLE:
            raise ImportError(
                "statsmodels is required for Fama-MacBeth regression. "
                "Install with: pip install statsmodels"
            )
        self.winsorise = winsorise
        self.nw_lags = nw_lags
        self.add_constant = add_constant

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_panel(
        self,
        mt_scores: pd.DataFrame,
        returns: pd.DataFrame,
        fundamentals: pd.DataFrame,
        ibes: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge all data sources into a clean monthly panel.

        Parameters
        ----------
        mt_scores:
            DataFrame with columns: [date, company_id, mt_score_strict,
            mt_score_lenient]. Date should be calendar month-end.
        returns:
            DataFrame with columns: [date, company_id, ret, mktcap, mktcap_lag].
            ``mktcap_lag`` = prior month market cap (for value-weighting and Size).
        fundamentals:
            DataFrame with columns: [date, company_id, book_equity].
            Used to compute log book-to-market ratio.
        ibes:
            DataFrame with columns: [date, company_id, sue].
            ``sue`` = standardised unexpected earnings.
        factors:
            Optional DataFrame with [date, mkt_rf, smb, hml, rmw, cma, umd]
            (Fama-French factors). If provided, factor returns are merged but
            not included as RHS variables (used only for time-series context).

        Returns
        -------
        pd.DataFrame: Monthly panel ready for :meth:`run_cross_sectional`.
                      Key columns:
                          date, company_id, ret_fwd, mt_score, size, logbm,
                          ret_1_0, ret_12_2, sue
        """
        # Standardise date columns to period-end timestamps
        mt = mt_scores.copy()
        ret = returns.copy()
        fund = fundamentals.copy()
        ib = ibes.copy()

        for df in [mt, ret, fund, ib]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        # Forward returns: ret_{t+1} is the dependent variable
        ret_fwd = ret[["date", "company_id", "ret"]].copy()
        ret_fwd["date"] = ret_fwd["date"] - pd.offsets.MonthEnd(1)
        ret_fwd = ret_fwd.rename(columns={"ret": "ret_fwd"})

        # Current returns (for momentum controls)
        ret_current = ret[["date", "company_id", "ret", "mktcap", "mktcap_lag"]].copy()

        # Compute Size = log(lagged market cap)
        ret_current["size"] = np.log(ret_current["mktcap_lag"].clip(lower=1))

        # Compute log BM
        fund["bm"] = fund["book_equity"] / fund.get("mktcap", np.nan)
        fund["logbm"] = np.log(fund["bm"].clip(lower=1e-6))

        # Merge panel
        panel = mt.merge(ret_fwd, on=["date", "company_id"], how="inner")
        panel = panel.merge(ret_current[["date", "company_id", "ret", "size"]],
                            on=["date", "company_id"], how="left")
        panel = panel.merge(fund[["date", "company_id", "logbm"]],
                            on=["date", "company_id"], how="left")
        panel = panel.merge(ib[["date", "company_id", "sue"]],
                            on=["date", "company_id"], how="left")

        if factors is not None and "date" in factors.columns:
            factors = factors.copy()
            factors["date"] = pd.to_datetime(factors["date"])
            panel = panel.merge(factors, on="date", how="left")

        # Compute momentum controls
        panel = self._add_momentum_controls(panel)

        # Winsorise
        if self.winsorise:
            cont_cols = ["mt_score_strict", "mt_score_lenient", "size", "logbm",
                         "ret_1_0", "ret_12_2", "sue"]
            panel = self._winsorise_panel(panel, cont_cols)

        logger.info(
            "prepare_panel: %d observations, %d firms, %d periods",
            len(panel),
            panel["company_id"].nunique(),
            panel["date"].nunique(),
        )
        return panel

    # ------------------------------------------------------------------
    # Cross-sectional regression
    # ------------------------------------------------------------------

    def run_cross_sectional(
        self,
        panel: pd.DataFrame,
        dependent_var: str = "ret_fwd",
        independent_vars: Optional[List[str]] = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Run month-by-month cross-sectional OLS regressions.

        For each period t, estimate:
            dependent_var_{i,t} = sum_k b_k_{t} * independent_var_k_{i,t} + e_{i,t}

        Parameters
        ----------
        panel:           Monthly panel from :meth:`prepare_panel`.
        dependent_var:   Name of the dependent variable column.
        independent_vars: List of RHS variable names. Defaults to MT + controls.
        date_col:        Period identifier column.

        Returns
        -------
        pd.DataFrame: Time series of period-t regression coefficients.
                      Index = dates; Columns = variable names (+ "const" if applicable).
        """
        if independent_vars is None:
            independent_vars = [
                "mt_score_strict", "size", "logbm", "ret_1_0", "ret_12_2", "sue"
            ]

        required = [dependent_var] + independent_vars
        missing = [c for c in required if c not in panel.columns]
        if missing:
            logger.warning("Missing columns in panel: %s", missing)
            independent_vars = [v for v in independent_vars if v in panel.columns]

        coeff_records: List[Dict[str, Any]] = []

        for period, period_df in panel.groupby(date_col):
            clean = period_df[[dependent_var] + independent_vars].dropna()

            # Require minimum observations for reliable estimation
            if len(clean) < max(20, len(independent_vars) + 5):
                continue

            y = clean[dependent_var].values
            X = clean[independent_vars].values
            if self.add_constant:
                X = sm.add_constant(X, has_constant="add")
                col_names = ["const"] + independent_vars
            else:
                col_names = independent_vars

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ols = sm.OLS(y, X).fit()
                record = {"date": period, "n_obs": len(clean)}
                for name, coef in zip(col_names, ols.params):
                    record[name] = coef
                coeff_records.append(record)
            except Exception as exc:
                logger.debug("OLS failed for period %s: %s", period, exc)

        if not coeff_records:
            return pd.DataFrame()

        coeff_df = pd.DataFrame(coeff_records).set_index("date").sort_index()
        logger.info(
            "run_cross_sectional: %d periods with valid regressions",
            len(coeff_df),
        )
        return coeff_df

    # ------------------------------------------------------------------
    # Fama-MacBeth statistics
    # ------------------------------------------------------------------

    def compute_fama_macbeth_stats(
        self,
        coefficient_series: pd.DataFrame,
        var_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute mean coefficients and Newey-West t-statistics.

        Pass 2 of the Fama-MacBeth procedure: time-series average the
        cross-sectional coefficients and apply Newey-West correction for
        serial correlation.

        Parameters
        ----------
        coefficient_series:
            Output of :meth:`run_cross_sectional` — time series of coefficients.
        var_names:
            Subset of variable names to report. Defaults to all columns except 'n_obs'.

        Returns
        -------
        pd.DataFrame: Rows = variable names; Columns = mean, nw_tstat, p_value, n_periods.
        """
        if var_names is None:
            var_names = [c for c in coefficient_series.columns if c != "n_obs"]

        results = []
        for var in var_names:
            if var not in coefficient_series.columns:
                continue
            series = coefficient_series[var].dropna()
            if len(series) < 2:
                continue

            mean_coef = float(series.mean())
            n_t = len(series)

            # Newey-West standard error
            try:
                nw_se = _newey_west_se(series.values, lags=self.nw_lags)
                t_stat = mean_coef / nw_se if nw_se > 0 else np.nan
            except Exception:
                # Fallback to simple t-stat
                t_stat = mean_coef / (series.std() / np.sqrt(n_t))
                nw_se = series.std() / np.sqrt(n_t)

            # Two-tailed p-value using t-distribution
            try:
                from scipy import stats as sp_stats
                p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=n_t - 1))
            except Exception:
                p_value = np.nan

            results.append(
                {
                    "variable": var,
                    "mean_coef": mean_coef,
                    "nw_se": float(nw_se),
                    "nw_tstat": float(t_stat),
                    "p_value": p_value,
                    "n_periods": n_t,
                }
            )

        return pd.DataFrame(results).set_index("variable")

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        panel: pd.DataFrame,
        mt_col: str = "mt_score_strict",
        controls: Optional[List[str]] = None,
        dependent_var: str = "ret_fwd",
    ) -> Dict[str, Any]:
        """
        Run the complete Fama-MacBeth analysis and return a summary dict.

        Runs three specifications:
            1. Univariate: MT only
            2. Multivariate: MT + all controls
            3. Multivariate with lenient MT: uses mt_score_lenient

        Parameters
        ----------
        panel:          Monthly panel from :meth:`prepare_panel`.
        mt_col:         Primary MT score column name.
        controls:       List of control variable names. Defaults to paper spec.
        dependent_var:  Dependent variable (next-period return).

        Returns
        -------
        Dict with keys:
            mt_alpha       : float — mean FM coefficient on MT (spec 2)
            mt_tstat       : float — Newey-West t-statistic on MT (spec 2)
            mt_p_value     : float — p-value
            full_results   : pd.DataFrame — full FM stats table (spec 2)
            univariate     : pd.DataFrame — univariate FM stats (spec 1)
            lenient        : pd.DataFrame — lenient MT FM stats (spec 3)
            n_periods      : int
            n_firms_avg    : float
        """
        if controls is None:
            controls = ["size", "logbm", "ret_1_0", "ret_12_2", "sue"]

        # Spec 1: univariate
        logger.info("Running univariate FM regression (MT only)…")
        coeff1 = self.run_cross_sectional(panel, dependent_var, [mt_col])
        stats1 = self.compute_fama_macbeth_stats(coeff1)

        # Spec 2: full multivariate
        indep_vars2 = [mt_col] + [c for c in controls if c in panel.columns]
        logger.info("Running multivariate FM regression (MT + controls)…")
        coeff2 = self.run_cross_sectional(panel, dependent_var, indep_vars2)
        stats2 = self.compute_fama_macbeth_stats(coeff2)

        # Spec 3: lenient MT
        lenient_col = "mt_score_lenient"
        stats3 = pd.DataFrame()
        if lenient_col in panel.columns:
            indep_vars3 = [lenient_col] + [c for c in controls if c in panel.columns]
            logger.info("Running FM regression with lenient MT score…")
            coeff3 = self.run_cross_sectional(panel, dependent_var, indep_vars3)
            stats3 = self.compute_fama_macbeth_stats(coeff3)

        # Extract headline MT stats from spec 2
        mt_alpha, mt_tstat, mt_pval = np.nan, np.nan, np.nan
        if mt_col in stats2.index:
            mt_alpha = float(stats2.loc[mt_col, "mean_coef"])
            mt_tstat = float(stats2.loc[mt_col, "nw_tstat"])
            mt_pval = float(stats2.loc[mt_col, "p_value"])

        n_periods = int(coeff2["n_obs"].count()) if "n_obs" in coeff2.columns else len(coeff2)
        n_firms_avg = float(coeff2["n_obs"].mean()) if "n_obs" in coeff2.columns else np.nan

        logger.info(
            "FM analysis complete: MT alpha=%.5f, t-stat=%.2f, p=%.4f",
            mt_alpha,
            mt_tstat,
            mt_pval,
        )

        return {
            "mt_alpha": mt_alpha,
            "mt_tstat": mt_tstat,
            "mt_p_value": mt_pval,
            "full_results": stats2,
            "univariate": stats1,
            "lenient": stats3,
            "coefficient_series": coeff2,
            "n_periods": n_periods,
            "n_firms_avg": n_firms_avg,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _winsorise_panel(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Winsorise continuous variables at [1%, 99%] cross-sectionally each period.

        Parameters
        ----------
        df:      Panel DataFrame with a ``date`` column.
        columns: List of column names to winsorise.

        Returns
        -------
        pd.DataFrame: DataFrame with winsorised values (in-place copy).
        """
        df = df.copy()

        def _win_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            for col in columns:
                if col in g.columns:
                    lo = g[col].quantile(_WIN_LOW)
                    hi = g[col].quantile(_WIN_HIGH)
                    g[col] = g[col].clip(lower=lo, upper=hi)
            return g

        df = df.groupby("date", group_keys=False).apply(_win_group)
        return df

    @staticmethod
    def _add_momentum_controls(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Ret(-1,0) and Ret(-12,-2) momentum controls from the ``ret`` column.

        This is a simplified approach assuming the panel already has the single-month
        return in ``ret``. For a production implementation, multi-period cumulative
        returns would be computed from a raw daily/monthly return file.

        Parameters
        ----------
        panel: Panel DataFrame with ``date``, ``company_id``, and ``ret``.

        Returns
        -------
        pd.DataFrame: Panel with ``ret_1_0`` and ``ret_12_2`` columns added.
        """
        if "ret" not in panel.columns:
            panel["ret_1_0"] = np.nan
            panel["ret_12_2"] = np.nan
            return panel

        panel = panel.sort_values(["company_id", "date"])
        panel["ret_1_0"] = panel.groupby("company_id")["ret"].shift(1)
        # 11-month cumulative return (months -12 to -2)
        panel["ret_12_2"] = (
            panel.groupby("company_id")["ret"]
            .transform(lambda x: x.shift(2).rolling(window=11).apply(
                lambda r: np.prod(1 + r) - 1, raw=True
            ))
        )
        return panel


# ---------------------------------------------------------------------------
# Newey-West standard error
# ---------------------------------------------------------------------------

def _newey_west_se(series: np.ndarray, lags: int = _NW_LAGS) -> float:
    """
    Compute Newey-West (HAC) standard error for a time series.

    Parameters
    ----------
    series: 1-D array of time-series values (regression coefficients).
    lags:   Number of autocorrelation lags to include.

    Returns
    -------
    float: Newey-West standard error.
    """
    n = len(series)
    demeaned = series - series.mean()
    # Variance
    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    nw_var = gamma_0

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma_l = float(np.dot(demeaned[lag:], demeaned[:-lag])) / n
        nw_var += 2 * weight * gamma_l

    nw_var = max(nw_var, 1e-12)  # numerical floor
    return float(np.sqrt(nw_var / n))
