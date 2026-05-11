"""
linkers.py
==========
Identifier linking tables (CCM, S&P 500 historical, CIQ company map).

Mirrors §4c, §4b and §9a of NB01 v2.

Tables used:
    crsp.ccmxpf_linktable  — gvkey ↔ permno (with linktype/linkprim filters
                              and date validity).
    comp.idxcst_his        — S&P 500 historical membership (gvkeyx='000003').
    ciq.wrds_gvkey         — gvkey ↔ Capital IQ companyid mapping.

The first two are re-exported from ``universe.py`` since the universe
builder already needs them. The third (CIQ) lives here because it is also
used by the transcripts module.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import wrds

# Re-export universe helpers so the public API of `linkers` stays stable.
from .universe import (  # noqa: F401 (re-export)
    fetch_ccm_link as fetch_ccm_linktable,
    fetch_sp500_history,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CIQ company map
# ---------------------------------------------------------------------------


def fetch_ciq_company_map(
    db: wrds.Connection,
    gvkeys: Iterable[str],
) -> pd.DataFrame:
    """Map gvkey → CIQ companyid via ``ciq.wrds_gvkey``.

    Returns a DataFrame with columns ``companyid, gvkey``. gvkeys are
    zero-padded to 6 characters.
    """
    gvkey_list = ",".join(f"'{str(g).zfill(6)}'" for g in gvkeys)
    if not gvkey_list:
        raise ValueError("gvkeys is empty.")
    query = f"""
        SELECT DISTINCT companyid, gvkey
        FROM ciq.wrds_gvkey
        WHERE gvkey IN ({gvkey_list})
    """
    logger.info("Pulling CIQ company map for %d gvkeys...",
                len(gvkey_list.split(",")))
    df = db.raw_sql(query)
    df.columns = [c.lower() for c in df.columns]
    logger.info("CIQ companies matched: %d / %d gvkeys",
                df["companyid"].nunique(), len(gvkey_list.split(",")))
    return df


def attach_ticker_to_ciq_map(
    ciq_map: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """Bring ticker into the CIQ map (NB01 v2 step 9a).

    Required columns:
        ciq_map  : companyid, gvkey
        universe : gvkey, ticker
    """
    out = ciq_map.merge(universe[["gvkey", "ticker"]], on="gvkey", how="left")
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_ciq_map(ciq_map: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "ciq_company_map.parquet"
    ciq_map.to_parquet(out, index=False)
    logger.info("Saved CIQ company map → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    """Standalone CLI: rebuild the CIQ map for the saved top-200 universe."""
    parser = argparse.ArgumentParser(description="Build CIQ company map for the top-200 universe.")
    parser.add_argument("--wrds_user", required=False)
    parser.add_argument("--raw_dir", default="data/raw",
                        help="Directory holding top200_universe.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    universe = pd.read_parquet(Path(args.raw_dir) / "top200_universe.parquet")
    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        ciq_map = fetch_ciq_company_map(db, universe["gvkey"].tolist())
        ciq_map = attach_ticker_to_ciq_map(ciq_map, universe)
        save_ciq_map(ciq_map, args.raw_dir)
    finally:
        db.close()


if __name__ == "__main__":
    _main()
