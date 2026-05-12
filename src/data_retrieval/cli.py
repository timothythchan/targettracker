"""Lightweight CLI for the WRDS data-retrieval pipeline.

This module intentionally avoids importing pandas/wrds at import time so
``--help`` works before runtime dependencies are installed.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence


DEFAULT_N_FIRMS = 200
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2023-12-31"
DEFAULT_RETURNS_END_DATE = "2024-12-31"


def build_parser() -> argparse.ArgumentParser:
    """Build the data-retrieval argument parser without heavy imports."""
    parser = argparse.ArgumentParser(description="Run the EarningsLens data-retrieval pipeline.")
    parser.add_argument(
        "--wrds_user",
        "--wrds-user",
        dest="wrds_user",
        required=False,
        help="WRDS username (uses .pgpass if omitted).",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="data/raw")
    parser.add_argument("--n_firms", "--n-firms", dest="n_firms", type=int, default=DEFAULT_N_FIRMS)
    parser.add_argument("--start_date", "--start-date", dest="start_date", default=DEFAULT_START_DATE)
    parser.add_argument("--end_date", "--end-date", dest="end_date", default=DEFAULT_END_DATE)
    parser.add_argument("--returns_end_date", "--returns-end-date", dest="returns_end_date", default=DEFAULT_RETURNS_END_DATE)
    parser.add_argument("--skip_sanity", "--skip-sanity", dest="skip_sanity", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the WRDS data-retrieval pipeline and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )

    try:
        from .pipeline import DataPipeline
    except ImportError as exc:
        parser.error(
            "data-retrieval dependencies are missing. Install them with "
            "`python -m pip install -r requirements.txt` before running the pipeline. "
            f"Original error: {exc}"
        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
