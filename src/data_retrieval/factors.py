"""
factors.py
==========
Retrieve Fama-French risk-factor data from WRDS.

Mirrors §8 of NB01 v2.

Tables used:
    ff.fivefactors_monthly — Fama-French 5-factor (monthly)
    ff.factors_monthly     — Fama-French 3-factor + momentum (for ``umd``)

Window: START_DATE → RETURNS_END_DATE (default 2010-01 → 2024-12).

Output: data/raw/ff_factors_monthly.parquet
        Columns: date, mktrf, smb, hml, rmw, cma, rf, umd
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import wrds

from .universe import RETURNS_END_DATE, START_DATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def fetch_ff5_with_momentum(
    db: wrds.Connection,
    start_date: str = START_DATE,
    end_date: str = RETURNS_END_DATE,
) -> pd.DataFrame:
    """Pull FF5 monthly + momentum (umd) joined on ``date``.

    Falls back to FF5 only if the join fails (older WRDS schemas).
    """
    join_q = f"""
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff.fivefactors_monthly AS f
        LEFT JOIN ff.factors_monthly AS m USING (date)
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    fallback_q = f"""
        SELECT date, mktrf, smb, hml, rmw, cma, rf
        FROM ff.fivefactors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    logger.info("Pulling Fama-French 5-factor + momentum (%s → %s)...",
                start_date, end_date)
    try:
        df = db.raw_sql(join_q, date_cols=["date"])
    except Exception as exc:  # noqa: BLE001 — WRDS schemas vary
        logger.warning("FF5+UMD join failed (%s); falling back to FF5-only.", exc)
        df = db.raw_sql(fallback_q, date_cols=["date"])
        df["umd"] = float("nan")
    logger.info("FF factors: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_factors(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ff_factors_monthly.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved FF factors → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pull Fama-French factors.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=RETURNS_END_DATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        ff = fetch_ff5_with_momentum(db, args.start_date, args.end_date)
        save_factors(ff, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    _main()
