"""
analyst_forecasts.py
====================
Retrieve IBES analyst forecast summary for benchmarking SUE.

Mirrors §7 of NB01 v2.

Table used:
    ibes.statsum_epsus — IBES US Summary Statistics (EPS, quarterly)

Filters (NB01 v2):
    fpi = '6'
    statpers BETWEEN START_DATE AND END_DATE

Columns retrieved:
    ticker, cusip, statpers, fpedats, fpi,
    numest, medest, meanest, stdev, actual

Output: data/raw/ibes_statsum.parquet

NOTE on fpi
-----------
The original v1 script used fpi='1' (one-quarter-ahead) per the standard
SUE literature (Livnat & Mendenhall 2006). NB01 v2 instead uses fpi='6'
to match the paper's chosen horizon. The CLI exposes ``--fpi`` so you can
re-pull the alternative without editing code.

SUE computation (kept for downstream evaluation):
    sue        = (actual - meanest) / stdev
    sue_scaled = (actual - meanest) / abs(prccq)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wrds

from .universe import END_DATE, START_DATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def fetch_ibes_summary(
    db: wrds.Connection,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    fpi: str = "6",
) -> pd.DataFrame:
    """Pull IBES quarterly summary statistics for the chosen forecast horizon.

    NB01 v2 uses ``fpi='6'`` (which in IBES summary tables corresponds to the
    quarterly forecast period it tracks). Pass ``fpi='1'`` for the classic
    one-quarter-ahead Livnat-Mendenhall horizon.
    """
    query = f"""
        SELECT i.ticker, i.cusip, i.statpers, i.fpedats, i.fpi,
               i.numest, i.medest, i.meanest, i.stdev, i.actual
        FROM ibes.statsum_epsus AS i
        WHERE i.fpi = '{fpi}'
          AND i.statpers BETWEEN '{start_date}' AND '{end_date}'
    """
    logger.info("Pulling IBES statsum_epsus (fpi=%s, %s → %s)...",
                fpi, start_date, end_date)
    df = db.raw_sql(query, date_cols=["statpers", "fpedats"])
    logger.info("IBES rows: %d · unique tickers: %d",
                len(df), df["ticker"].nunique())
    return df


# ---------------------------------------------------------------------------
# SUE computation (downstream helper)
# ---------------------------------------------------------------------------


def compute_sue(
    ibes: pd.DataFrame,
    fundq: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute Standardised Unexpected Earnings (SUE).

    sue        = (actual - meanest) / stdev
    sue_scaled = (actual - meanest) / abs(prccq)   (if fundq supplied)
    """
    out = ibes.copy()
    diff = out["actual"] - out["meanest"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sue"] = np.where(out["stdev"] > 0, diff / out["stdev"], np.nan)

    if fundq is not None and "prccq" in fundq.columns and "ticker" in fundq.columns:
        merge_cols = ["ticker", "datadate", "prccq"]
        out = out.merge(
            fundq[merge_cols].rename(columns={"datadate": "fpedats"}),
            on=["ticker", "fpedats"],
            how="left",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sue_scaled"] = np.where(
                out["prccq"].abs() > 0, diff / out["prccq"].abs(), np.nan
            )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_analyst_forecasts(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ibes_statsum.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved IBES summary → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pull IBES analyst forecasts.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=END_DATE)
    parser.add_argument("--fpi", default="6")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        ibes = fetch_ibes_summary(db, args.start_date, args.end_date, fpi=args.fpi)
        save_analyst_forecasts(ibes, args.output_dir)
    finally:
        db.close()


if __name__ == "__main__":
    _main()
