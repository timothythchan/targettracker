"""
fundamentals.py
===============
Retrieve Compustat quarterly fundamentals for the **top-200 universe**.

Mirrors §6 of NB01 v2.

Table used:
    comp.fundq — Compustat Quarterly Fundamentals

Filters:
    indfmt='INDL', datafmt='STD', popsrc='D', consol='C'
    gvkey IN (top-200 gvkeys)
    datadate BETWEEN START_DATE AND END_DATE

Columns retrieved (NB01 v2 list):
    gvkey, datadate, fyearq, fqtr, rdq, conm,
    saleq, revtq, cogsq, oibdpq, niq, epspxq, epsfxq,
    atq, ltq, cheq, dlcq, dlttq, ceqq, seqq,
    cshoq, prccq

Output: data/raw/compustat_fundq.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import wrds

from .universe import END_DATE, START_DATE, build_top_n_universe, fetch_crsp_monthly_mcap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def fetch_quarterly_fundamentals(
    db: wrds.Connection,
    gvkeys: Iterable[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Pull comp.fundq filtered to the top-200 gvkeys.

    gvkeys are zero-padded to 6 characters before being placed in the SQL
    IN-list (Compustat convention).
    """
    gvkey_list = ",".join(f"'{str(g).zfill(6)}'" for g in gvkeys)
    if not gvkey_list:
        raise ValueError("gvkeys is empty — pass the top-200 gvkey list.")

    query = f"""
        SELECT gvkey, datadate, fyearq, fqtr, rdq, conm,
               saleq, revtq, cogsq, oibdpq, niq, epspxq, epsfxq,
               atq, ltq, cheq, dlcq, dlttq, ceqq, seqq,
               cshoq, prccq
        FROM comp.fundq
        WHERE gvkey IN ({gvkey_list})
          AND indfmt = 'INDL' AND datafmt = 'STD'
          AND popsrc  = 'D'   AND consol  = 'C'
          AND datadate BETWEEN '{start_date}' AND '{end_date}'
    """
    n_gvkeys = len(gvkey_list.split(","))
    logger.info("Pulling Compustat fundq for %d gvkeys (%s → %s)...",
                n_gvkeys, start_date, end_date)
    df = db.raw_sql(query, date_cols=["datadate", "rdq"])
    logger.info("Compustat fundq: %d rows · %d unique gvkeys",
                len(df), df["gvkey"].nunique())
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_fundamentals(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "compustat_fundq.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved fundamentals → %s (%.1f MB)",
                out, out.stat().st_size / 1e6)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pull Compustat fundamentals for top-200 universe.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=END_DATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        crsp_m = fetch_crsp_monthly_mcap(db)
        universe = build_top_n_universe(db, crsp_monthly=crsp_m)
        fundq = fetch_quarterly_fundamentals(db, universe["gvkey"].tolist(),
                                             start_date=args.start_date,
                                             end_date=args.end_date)
        save_fundamentals(fundq, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    _main()
