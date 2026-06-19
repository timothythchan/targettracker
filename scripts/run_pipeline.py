#!/usr/bin/env python
"""
run_pipeline.py — One-shot orchestrator for the full notebook-free pipeline.

Runs each stage in order (subject to ``--start`` / ``--stop``):

    1. data         scripts/run_data_retrieval.py            (NB01)
    2. baseline     scripts/run_spacy_baseline.py            (NB02)
    3. llm          scripts/run_llm_extraction.py            (NB03)
    4. rag          scripts/run_rag_matching.py              (NB04)
    5. calibrate    scripts/run_threshold_calibration.py     (NB04b)
    6. demo         scripts/build_demo_cache.py              (NB06)

Each stage is a separate subprocess so a failure in one does not poison
the next stage's interpreter state.

Stages that need credentials (data via WRDS, llm via an LLM API key) will
fail loudly when those credentials are missing. Use ``--skip data llm``
or ``--start rag`` to short-circuit them when reusing pre-built parquet
files.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --start rag             # reuse llm_targets.parquet
    python scripts/run_pipeline.py --skip llm calibrate    # quick rebuild
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

logger = logging.getLogger("scripts.run_pipeline")


STAGES = [
    ("data",      "run_data_retrieval.py",        "NB01 - WRDS data pull"),
    ("baseline",  "run_spacy_baseline.py",        "NB02 - spaCy baseline + MT"),
    ("llm",       "run_llm_extraction.py",        "NB03 - LLM target extraction"),
    ("rag",       "run_rag_matching.py",          "NB04 - semantic MT batch"),
    ("calibrate", "run_threshold_calibration.py", "NB04b - threshold calibration"),
    ("demo",      "build_demo_cache.py",          "NB06 - Gradio demo cache"),
]
STAGE_ORDER = [name for name, _, _ in STAGES]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--start",
        choices=STAGE_ORDER,
        default="data",
        help="First stage to run (default: data).",
    )
    p.add_argument(
        "--stop",
        choices=STAGE_ORDER,
        default="demo",
        help="Last stage to run, inclusive (default: demo).",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=STAGE_ORDER,
        help="Stages to skip even when within --start/--stop range.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to subsequent stages even if a stage fails. Default is "
            "to abort on the first non-zero return code."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_idx = STAGE_ORDER.index(args.start)
    stop_idx = STAGE_ORDER.index(args.stop)
    if stop_idx < start_idx:
        raise SystemExit("--stop must not precede --start")

    skipped = set(args.skip)
    failures: List[str] = []

    for name, script, desc in STAGES[start_idx : stop_idx + 1]:
        if name in skipped:
            logger.info("[skip] %s — %s", name, desc)
            continue
        cmd = [sys.executable, str(SCRIPTS_DIR / script)]
        logger.info("[run ] %s — %s", name, desc)
        logger.info("       %s", " ".join(cmd))
        if args.dry_run:
            continue

        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
        if rc == 0:
            logger.info("[ ok ] %s exited 0", name)
        else:
            logger.error("[fail] %s exited %d", name, rc)
            failures.append(name)
            if not args.continue_on_error:
                return rc

    if failures:
        logger.error("Failures in stages: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
