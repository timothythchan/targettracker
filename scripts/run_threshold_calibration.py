#!/usr/bin/env python
"""
run_threshold_calibration.py — Calibrate the RAG cosine thresholds against a
human-labeled gold sample (port of NB04b ``04b_threshold_calibration.ipynb``).

Inputs
------
- ``data/processed/mt_calibration_sample_labeled.csv`` (required)

Outputs
-------
- ``data/processed/mt_calibration_result.json``
- ``data/processed/semantic_mt_scores_calibrated.meta.json``

The output JSON is consumed by ``scripts/run_rag_matching.py --calibration``
and by ``demo/app.py`` indirectly (via the matcher used inside the LangGraph
pipeline).

All of the heavy lifting lives in ``src.rag.calibration``; this script is
just a thin CLI wrapper that wires the file paths together and forwards
the optional knobs (confidence-weight scheme, bootstrap iterations).

Usage
-----
    python scripts/run_threshold_calibration.py
    python scripts/run_threshold_calibration.py \
        --labeled data/processed/mt_calibration_sample_labeled.csv \
        --bootstrap 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("scripts.run_threshold_calibration")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_threshold_calibration",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Project data root (default: ./data).",
    )
    p.add_argument(
        "--labeled",
        type=Path,
        default=None,
        help=(
            "Path to mt_calibration_sample_labeled.csv. Defaults to "
            "<data-dir>/processed/mt_calibration_sample_labeled.csv."
        ),
    )
    p.add_argument(
        "--result-out",
        type=Path,
        default=None,
        help=(
            "Output JSON path for mt_calibration_result.json. Defaults to "
            "<data-dir>/processed/mt_calibration_result.json."
        ),
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help=(
            "Output JSON path for semantic_mt_scores_calibrated.meta.json. "
            "Defaults to <data-dir>/processed/."
        ),
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations for the F1-optimal CI (default: 1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260509,
        help="RNG seed for the bootstrap (default: 20260509, matches NB04b).",
    )
    p.add_argument(
        "--confidence-weight",
        action="append",
        metavar="LEVEL=WEIGHT",
        help=(
            "Override the {H, M, L} confidence weight scheme. Pass once per "
            "level, e.g. --confidence-weight H=1.0 --confidence-weight M=0.5 "
            "--confidence-weight L=0.0."
        ),
    )
    p.add_argument(
        "--placeholder-maintained",
        type=float,
        default=0.65,
        help="v1 placeholder for the maintained boundary (default: 0.65).",
    )
    p.add_argument(
        "--placeholder-rephrased",
        type=float,
        default=0.40,
        help="v1 placeholder for the rephrased boundary (default: 0.40).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _parse_weights(specs: Optional[List[str]]) -> dict:
    if not specs:
        from src.rag.calibration import DEFAULT_CONFIDENCE_WEIGHTS
        return dict(DEFAULT_CONFIDENCE_WEIGHTS)

    weights = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--confidence-weight value '{spec}' must be LEVEL=WEIGHT")
        level, raw = spec.split("=", 1)
        weights[level.strip().upper()] = float(raw)
    for key in ("H", "M", "L"):
        if key not in weights:
            raise SystemExit(f"--confidence-weight: missing level '{key}'")
    return weights


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_processed = args.data_dir / "processed"
    labeled_csv = args.labeled or (data_processed / "mt_calibration_sample_labeled.csv")
    result_out = args.result_out or (data_processed / "mt_calibration_result.json")
    meta_out = args.meta_out or (data_processed / "semantic_mt_scores_calibrated.meta.json")

    weights = _parse_weights(args.confidence_weight)
    logger.info("Confidence weights: %s", weights)

    from src.rag.calibration import (
        load_labels,
        run_calibration,
        save_calibration_result,
    )

    df = load_labels(labeled_csv)
    result = run_calibration(
        df,
        confidence_weights=weights,
        placeholder_maintained=args.placeholder_maintained,
        placeholder_rephrased=args.placeholder_rephrased,
        bootstrap_iter=args.bootstrap,
        bootstrap_seed=args.seed,
    )

    save_calibration_result(
        result,
        result_path=result_out,
        meta_path=meta_out,
        labeled_csv=labeled_csv,
    )

    m = result["maintained_threshold"]
    t = result["tracked_threshold"]
    logger.info("=" * 70)
    logger.info("CALIBRATION SUMMARY")
    logger.info("  maintained: F1-opt=%.3f | placeholder=%.3f | AUC=%.3f | n=%d",
                m["calibrated_f1"], m["placeholder"], m["auc"], m["n"])
    logger.info("  tracked   : F1-opt=%.3f | placeholder=%.3f | AUC=%.3f | n=%d",
                t["calibrated_f1"], t["placeholder"], t["auc"], t["n"])
    ci = result["bootstrap_ci_95"]
    logger.info("  bootstrap CI (95%%): maintained=%s, tracked=%s",
                ci["maintained"], ci["tracked"])
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
