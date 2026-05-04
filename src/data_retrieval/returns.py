"""
returns.py
==========
Retrieve CRSP equity return data for the **top-200 universe**.

Mirrors §5 of NB01 v2.

Tables used:
    crsp.dsf  — Daily Stock File   (daily returns for CAR computation)
    crsp.msf  — Monthly Stock File (already pulled by universe.py;
                                    we just subset & save it here)

Filters:
    permno IN (top-200 permnos)
    date BETWEEN START_DATE AND RETURNS_END_DATE

Date range:
    START_DATE       = 2010-01-01
    RETURNS_END_DATE = 2024-12-31  (extend 1y past END_DATE for forward returns)

Outputs:
    data/raw/crsp_daily.parquet
    data/raw/crsp_monthly.parquet  (subset of the monthly pull from universe.py)

Usage (programmatic):
    from data_retrieval.universe import build_top_n_universe, fetch_crsp_monthly_mcap
    from data_retrieval.returns import fetch_daily_returns, save_returns
    import wrds
    db = wrds.Connection()
    crsp_m = fetch_crsp_monthly_mcap(db)
    universe = build_top_n_universe(db, crsp_monthly=crsp_m)
    crsp_d   = fetch_daily_returns(db, universe["permno"].tolist())
    save_returns(crsp_d, crsp_m, universe, "data/raw")

Usage (standalone CLI):
    python -m data_retrieval.returns --wrds_user your_username --output_dir data/raw
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import wrds

from .universe import (
    END_DATE,
    RETURNS_END_DATE,
    START_DATE,
    build_top_n_universe,
    fetch_crsp_monthly_mcap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily returns (filtered to top-200 permnos)
# ---------------------------------------------------------------------------


def fetch_daily_returns(
    db: wrds.Connection,
    permnos: Iterable[int],
    start_date: str = START_DATE,
    end_date: str = RETURNS_END_DATE,
) -> pd.DataFrame:
    """Pull CRSP daily for the given permnos.

    Returns columns:
        permno, date, ret, retx, prc, vol, shrout, mktcap_d
    """
    permno_list = ",".join(str(int(p)) for p in permnos)
    if not permno_list:
        raise ValueError("permnos is empty — pass the top-200 permno list.")

    query = f"""
        SELECT permno, date, ret, retx, prc, vol, shrout,
               ABS(prc) * shrout / 1000 AS mktcap_d
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '{start_date}' AND '{end_date}'
    """
    n_permnos = len(permno_list.split(","))
    logger.info("Pulling CRSP daily for %d permnos (%s → %s)...",
                n_permnos, start_date, end_date)
    df = db.raw_sql(query, date_cols=["date"])
    logger.info("CRSP daily: %d rows (~%.0f obs/firm)",
                len(df), len(df) / n_permnos)
    return df


# ---------------------------------------------------------------------------
# Monthly subset
# ---------------------------------------------------------------------------


def subset_monthly_to_universe(
    crsp_monthly: pd.DataFrame, permnos: Iterable[int]
) -> pd.DataFrame:
    """Filter the full CRSP monthly pull down to the top-200 permnos."""
    permnos = set(int(p) for p in permnos)
    out = crsp_monthly[crsp_monthly["permno"].isin(permnos)].copy()
    logger.info("CRSP monthly (top-200 subset): %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_returns(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    universe: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save crsp_daily.parquet + crsp_monthly.parquet (top-200 subset)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_top = subset_monthly_to_universe(monthly, universe["permno"])

    daily_path   = output_dir / "crsp_daily.parquet"
    monthly_path = output_dir / "crsp_monthly.parquet"
    daily.to_parquet(daily_path, index=False)
    monthly_top.to_parquet(monthly_path, index=False)

    logger.info("Saved CRSP daily   → %s (%.1f MB)",
                daily_path, daily_path.stat().st_size / 1e6)
    logger.info("Saved CRSP monthly → %s (%.1f MB)",
                monthly_path, monthly_path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pull CRSP daily/monthly for the top-200 universe.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=RETURNS_END_DATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        crsp_m = fetch_crsp_monthly_mcap(db, start_date=args.start_date, end_date=args.end_date)
        universe = build_top_n_universe(db, crsp_monthly=crsp_m)
        crsp_d = fetch_daily_returns(db, universe["permno"].tolist(),
                                     start_date=args.start_date, end_date=args.end_date)
        save_returns(crsp_d, crsp_m, universe, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    _main()
