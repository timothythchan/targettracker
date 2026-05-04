"""
universe.py
===========
Build the **Top-200 S&P 500 universe** used by every downstream pull in
the EarningsLens pipeline.  Mirrors §4 of NB01 v2 exactly.

Methodology
-----------
1. Pull CRSP monthly stock file (``crsp.msf``) joined to ``crsp.msenames``
   with monthly market cap, restricted to NYSE/AMEX/NASDAQ ordinary
   common shares (``exchcd IN (1,2,3)``, ``shrcd IN (10,11)``).
2. Pull S&P 500 historical membership from ``comp.idxcst_his``
   (``gvkeyx = '000003'``).
3. Map gvkey ↔ permno via ``crsp.ccmxpf_linktable``
   (``linktype IN ('LU','LC')``, ``linkprim IN ('P','C')``).
4. Restrict S&P 500 members to those active during the ranking window.
5. Compute average monthly market cap per permno over the ranking
   window; require at least 36 monthly observations for stable
   ranking.
6. Take the top ``N_FIRMS`` (default 200) by average market cap.

Output: a parquet file containing the universe with columns
``permno, gvkey, ticker, comnam, avg_mcap, n_months``.

Constants match NB01 v2:
    START_DATE = 2010-01-01
    END_DATE   = 2023-12-31     (ranking window end)
    RETURNS_END_DATE = 2024-12-31  (extend returns 1y for forward windows)
    N_FIRMS = 200

Usage (programmatic):
    from data_retrieval.universe import build_top_n_universe, save_universe
    import wrds
    db = wrds.Connection()
    universe = build_top_n_universe(db, n_firms=200)
    save_universe(universe, "data/raw")

Usage (standalone CLI):
    python -m data_retrieval.universe --wrds_user your_username \\
        --output_dir data/raw --n_firms 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import wrds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror NB01 v2 §2 ("Pipeline Configuration")
# ---------------------------------------------------------------------------

START_DATE: str = "2010-01-01"
END_DATE: str = "2023-12-31"
RETURNS_END_DATE: str = "2024-12-31"

RANKING_START: str = "2010-01-01"
RANKING_END: str = "2023-12-31"

N_FIRMS: int = 200
MIN_MONTHS: int = 36  # require ≥3y of CRSP data for stable ranking

EXCHCD: Tuple[int, ...] = (1, 2, 3)   # NYSE, AMEX, NASDAQ
SHRCD: Tuple[int, ...] = (10, 11)     # ordinary common shares
SP500_GVKEYX: str = "000003"          # S&P 500 index gvkeyx in idxcst_his


# ---------------------------------------------------------------------------
# CRSP monthly with market cap
# ---------------------------------------------------------------------------


def fetch_crsp_monthly_mcap(
    db: wrds.Connection,
    start_date: str = START_DATE,
    end_date: str = RETURNS_END_DATE,
) -> pd.DataFrame:
    """Pull CRSP monthly stock file with market cap and exchange info.

    This is the heaviest single query in the pipeline (≈2-3 min on WRDS).
    The result is reused by the universe builder *and* persisted as
    ``crsp_monthly.parquet`` (filtered to the top-200 permnos) by the
    returns module — so we don't re-pull it.
    """
    exchcd_list = ",".join(str(x) for x in EXCHCD)
    shrcd_list = ",".join(str(x) for x in SHRCD)

    query = f"""
        SELECT a.permno, a.permco, a.date, a.ret, a.shrout, a.prc,
               ABS(a.prc) * a.shrout / 1000 AS mktcap_m,
               b.ticker, b.comnam, b.exchcd, b.shrcd
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
          ON a.permno = b.permno
         AND b.namedt <= a.date
         AND a.date  <= b.nameendt
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND b.exchcd IN ({exchcd_list})
          AND b.shrcd  IN ({shrcd_list})
    """
    logger.info("Pulling CRSP monthly with market cap (%s → %s)...", start_date, end_date)
    df = db.raw_sql(query, date_cols=["date"])
    logger.info("CRSP monthly: %d rows · %d unique permnos",
                len(df), df["permno"].nunique())
    return df


# ---------------------------------------------------------------------------
# S&P 500 membership
# ---------------------------------------------------------------------------


def fetch_sp500_history(db: wrds.Connection) -> pd.DataFrame:
    """Pull S&P 500 historical membership from ``comp.idxcst_his``.

    Columns: gvkey, iid, gvkeyx, sp_from, sp_thru.
    """
    query = f"""
        SELECT gvkey, iid, gvkeyx, "from" AS sp_from, thru AS sp_thru
        FROM comp.idxcst_his
        WHERE gvkeyx = '{SP500_GVKEYX}'
    """
    logger.info("Pulling S&P 500 historical membership...")
    df = db.raw_sql(query, date_cols=["sp_from", "sp_thru"])
    logger.info("S&P 500 ever-members: %d", df["gvkey"].nunique())
    return df


# ---------------------------------------------------------------------------
# CCM link
# ---------------------------------------------------------------------------


def fetch_ccm_link(db: wrds.Connection) -> pd.DataFrame:
    """Pull the CRSP/Compustat link table (filtered to LU/LC, P/C)."""
    query = """
        SELECT gvkey, lpermno AS permno, lpermco AS permco, linktype, linkprim,
               linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """
    logger.info("Pulling CRSP/Compustat link table...")
    df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    logger.info("CCM link rows: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------


def build_top_n_universe(
    db: wrds.Connection,
    n_firms: int = N_FIRMS,
    ranking_start: str = RANKING_START,
    ranking_end: str = RANKING_END,
    min_months: int = MIN_MONTHS,
    crsp_monthly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the top-N S&P 500 universe by average market cap.

    Parameters
    ----------
    db : wrds.Connection
        Open WRDS connection.
    n_firms : int
        Number of firms to keep (default 200).
    ranking_start, ranking_end : str
        Window over which we average market cap.
    min_months : int
        Minimum CRSP monthly observations within the ranking window for
        a permno to be eligible (default 36 = 3 years).
    crsp_monthly : pd.DataFrame, optional
        Pre-fetched CRSP monthly with market cap (avoids re-pulling the
        2–3 min query when the caller already has it).

    Returns
    -------
    pd.DataFrame with columns:
        permno, gvkey, ticker, comnam, avg_mcap, n_months
    sorted by avg_mcap descending.
    """
    if crsp_monthly is None:
        crsp_monthly = fetch_crsp_monthly_mcap(db)
    sp500_hist = fetch_sp500_history(db)
    link = fetch_ccm_link(db)

    # Filter S&P 500 members active in the ranking window
    sp_in_window = sp500_hist[
        (sp500_hist["sp_from"] <= pd.Timestamp(ranking_end))
        & (
            (sp500_hist["sp_thru"].isna())
            | (sp500_hist["sp_thru"] >= pd.Timestamp(ranking_start))
        )
    ].copy()
    logger.info("S&P 500 members during %s..%s: %d",
                ranking_start, ranking_end, sp_in_window["gvkey"].nunique())

    # Map gvkey → permno (date-valid)
    sp_with_permno = (
        sp_in_window.merge(
            link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey"
        )
        .query("linkdt <= @ranking_end and linkenddt >= @ranking_start")
        [["gvkey", "permno"]]
        .drop_duplicates()
    )
    logger.info("S&P 500 unique permnos with link: %d",
                sp_with_permno["permno"].nunique())

    # Compute avg market cap over the ranking window for S&P 500 permnos
    mcap = (
        crsp_monthly.query("@ranking_start <= date <= @ranking_end")
        .merge(sp_with_permno, on="permno", how="inner")
        .groupby("permno")
        .agg(
            avg_mcap=("mktcap_m", "mean"),
            n_months=("mktcap_m", "size"),
            ticker=("ticker", "last"),
            comnam=("comnam", "last"),
        )
        .reset_index()
    )

    # Stability filter
    mcap_stable = (
        mcap.query("n_months >= @min_months")
        .sort_values("avg_mcap", ascending=False)
    )

    top_n = mcap_stable.head(n_firms).copy()
    top_n = top_n.merge(sp_with_permno, on="permno", how="left")
    # Normalise gvkey to zero-padded 6-character strings (Compustat convention)
    top_n["gvkey"] = top_n["gvkey"].astype(str).str.zfill(6)
    top_n["permno"] = top_n["permno"].astype(int)

    logger.info("Top-%d universe built. Median avg_mcap = $%.1f B",
                n_firms, top_n["avg_mcap"].median() / 1000)

    return top_n.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_universe(universe: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save the universe table as ``top200_universe.parquet`` (NB01 v2 name)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "top200_universe.parquet"
    universe.to_parquet(out, index=False)
    logger.info("Saved universe → %s (%d firms)", out, len(universe))
    return out


def load_universe(raw_dir: str | Path) -> pd.DataFrame:
    """Re-load the saved universe parquet."""
    return pd.read_parquet(Path(raw_dir) / "top200_universe.parquet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the EarningsLens top-N universe.")
    parser.add_argument("--wrds_user", required=False,
                        help="WRDS username (uses .pgpass if omitted).")
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--n_firms", type=int, default=N_FIRMS)
    parser.add_argument("--ranking_start", default=RANKING_START)
    parser.add_argument("--ranking_end", default=RANKING_END)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    db = wrds.Connection(wrds_username=args.wrds_user) if args.wrds_user else wrds.Connection()
    try:
        universe = build_top_n_universe(
            db,
            n_firms=args.n_firms,
            ranking_start=args.ranking_start,
            ranking_end=args.ranking_end,
        )
        save_universe(universe, args.output_dir)
        print(universe.head(10).to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    _main()
