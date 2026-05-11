"""
pipeline.py
===========
Master orchestrator for the EarningsLens data-retrieval pipeline.

End-to-end mirror of NB01 v2:

    §4  Build top-200 universe (S&P 500 historical members ranked by avg
        market cap over 2010-2023, ≥36 months coverage).
    §5  Pull CRSP daily and subset CRSP monthly to the 200 permnos.
    §6  Pull Compustat fundq (quarterly fundamentals) for the 200 gvkeys.
    §7  Pull IBES analyst summary (fpi='6').
    §8  Pull Fama-French 5-factor + momentum.
    §9  Pull CIQ earnings-call transcripts (metadata, then components in
        batches of 500), join, and save the canonical ciq_transcripts.parquet.
    §10 Sanity checks (file sizes, coverage assertions).

Public class
------------
``DataPipeline`` manages a single WRDS connection lifecycle and runs all
modules in sequence.  It can also be used as a context manager::

    with DataPipeline(wrds_username="your_username") as pipe:
        pipe.run_all(output_dir="data/raw")
        pipe.sanity_check(raw_dir="data/raw")

Usage (standalone CLI):
    python -m data_retrieval.pipeline --wrds_user your_username --output_dir data/raw
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import wrds

from .analyst_forecasts import fetch_ibes_summary, save_analyst_forecasts
from .factors import fetch_ff5_with_momentum, save_factors
from .fundamentals import fetch_quarterly_fundamentals, save_fundamentals
from .returns import fetch_daily_returns, save_returns
from .transcripts import build_full_transcript_dataset
from .universe import (
    END_DATE,
    N_FIRMS,
    RANKING_END,
    RANKING_START,
    RETURNS_END_DATE,
    START_DATE,
    build_top_n_universe,
    fetch_crsp_monthly_mcap,
    save_universe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Run the full top-200 EarningsLens data-retrieval pipeline."""

    def __init__(
        self,
        wrds_username: Optional[str] = None,
        n_firms: int = N_FIRMS,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        returns_end_date: str = RETURNS_END_DATE,
        ranking_start: str = RANKING_START,
        ranking_end: str = RANKING_END,
    ) -> None:
        self.wrds_username = wrds_username
        self.n_firms = n_firms
        self.start_date = start_date
        self.end_date = end_date
        self.returns_end_date = returns_end_date
        self.ranking_start = ranking_start
        self.ranking_end = ranking_end
        self.db: Optional[wrds.Connection] = None

        # Cached datasets to avoid duplicate WRDS pulls
        self._crsp_monthly: Optional[pd.DataFrame] = None
        self._universe: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> wrds.Connection:
        if self.db is None:
            self.db = (
                wrds.Connection(wrds_username=self.wrds_username)
                if self.wrds_username
                else wrds.Connection()
            )
            logger.info("WRDS connection opened.")
        return self.db

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
                logger.info("WRDS connection closed.")
            finally:
                self.db = None

    def __enter__(self) -> "DataPipeline":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def fetch_universe(self) -> pd.DataFrame:
        """Build (and cache) the top-N universe."""
        if self._universe is not None:
            return self._universe

        db = self.connect()
        if self._crsp_monthly is None:
            self._crsp_monthly = fetch_crsp_monthly_mcap(
                db, start_date=self.start_date, end_date=self.returns_end_date
            )
        self._universe = build_top_n_universe(
            db,
            n_firms=self.n_firms,
            ranking_start=self.ranking_start,
            ranking_end=self.ranking_end,
            crsp_monthly=self._crsp_monthly,
        )
        return self._universe

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------

    def run_all(self, output_dir: str | Path) -> None:
        """Run §4 → §9 in order, persisting every intermediate parquet."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        db = self.connect()

        # §4 — universe
        universe = self.fetch_universe()
        save_universe(universe, output_dir)

        # §5 — returns (we already have the monthly pull cached)
        crsp_d = fetch_daily_returns(
            db, universe["permno"].tolist(),
            start_date=self.start_date, end_date=self.returns_end_date,
        )
        save_returns(crsp_d, self._crsp_monthly, universe, output_dir)

        # §6 — fundamentals
        fundq = fetch_quarterly_fundamentals(
            db, universe["gvkey"].tolist(),
            start_date=self.start_date, end_date=self.end_date,
        )
        save_fundamentals(fundq, output_dir)

        # §7 — analyst forecasts
        ibes = fetch_ibes_summary(
            db, start_date=self.start_date, end_date=self.end_date, fpi="6"
        )
        save_analyst_forecasts(ibes, output_dir)

        # §8 — Fama-French factors
        ff = fetch_ff5_with_momentum(
            db, start_date=self.start_date, end_date=self.returns_end_date
        )
        save_factors(ff, output_dir)

        # §9 — transcripts (map + meta + batched components + assemble)
        build_full_transcript_dataset(
            db, universe, output_dir,
            start_date=self.start_date, end_date=self.end_date,
        )

        logger.info("DataPipeline.run_all complete. Output → %s", output_dir)

    # ------------------------------------------------------------------
    # Sanity checks (NB01 v2 §10)
    # ------------------------------------------------------------------

    def sanity_check(self, raw_dir: str | Path) -> None:
        """Run NB01 v2's coverage assertions on the persisted parquets."""
        raw_dir = Path(raw_dir)

        files = [
            ("top200_universe.parquet",     "Top-200 firm universe"),
            ("crsp_daily.parquet",          "CRSP daily returns"),
            ("crsp_monthly.parquet",        "CRSP monthly returns"),
            ("compustat_fundq.parquet",     "Compustat fundamentals (quarterly)"),
            ("ibes_statsum.parquet",        "IBES analyst forecasts"),
            ("ff_factors_monthly.parquet",  "Fama-French factors"),
            ("ciq_company_map.parquet",     "CIQ ticker → companyid map"),
            ("ciq_transcript_meta.parquet", "Transcript metadata"),
            ("ciq_transcripts.parquet",     "Transcript components (full text)"),
        ]
        total_mb = 0.0
        print("=" * 60)
        print("DATA RETRIEVAL SUMMARY")
        print("=" * 60)
        for fname, desc in files:
            fp = raw_dir / fname
            if fp.exists():
                size_mb = fp.stat().st_size / 1e6
                total_mb += size_mb
                print(f"  ✓ {fname:35s} {size_mb:8.1f} MB  · {desc}")
            else:
                print(f"  ✗ {fname:35s}      MISSING  · {desc}")
        print(f"\nTotal raw data : {total_mb:,.1f} MB")

        # Hard assertions
        universe_df = pd.read_parquet(raw_dir / "top200_universe.parquet")
        crsp_d_df = pd.read_parquet(raw_dir / "crsp_daily.parquet",
                                    columns=["permno", "date"])
        trans_df = pd.read_parquet(raw_dir / "ciq_transcripts.parquet",
                                   columns=["ticker", "transcriptid", "event_date"])

        assert len(universe_df) == self.n_firms, (
            f"Universe size {len(universe_df)} ≠ {self.n_firms}"
        )
        assert crsp_d_df["permno"].nunique() >= int(self.n_firms * 0.95), (
            "CRSP coverage <95% of universe"
        )
        unique_transcripts = trans_df["transcriptid"].nunique()
        assert unique_transcripts >= 5000, (
            f"Only {unique_transcripts} transcripts — too few, expected ~11,000"
        )
        per_firm = trans_df.groupby("ticker")["transcriptid"].nunique().describe()
        print("\nTranscripts per firm (distribution):")
        print(per_firm.to_string())
        print(
            f"\n✓ All sanity checks passed. "
            f"{unique_transcripts:,} transcripts across "
            f"{trans_df['ticker'].nunique()} tickers."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the EarningsLens data-retrieval pipeline.")
    parser.add_argument("--wrds_user", required=False,
                        help="WRDS username (uses .pgpass if omitted).")
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--n_firms", type=int, default=N_FIRMS)
    parser.add_argument("--start_date", default=START_DATE)
    parser.add_argument("--end_date", default=END_DATE)
    parser.add_argument("--returns_end_date", default=RETURNS_END_DATE)
    parser.add_argument("--skip_sanity", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s")

    with DataPipeline(
        wrds_username=args.wrds_user,
        n_firms=args.n_firms,
        start_date=args.start_date,
        end_date=args.end_date,
        returns_end_date=args.returns_end_date,
    ) as pipe:
        pipe.run_all(args.output_dir)
        if not args.skip_sanity:
            pipe.sanity_check(args.output_dir)


if __name__ == "__main__":
    _main()
