#!/usr/bin/env python
"""
run_rag_matching.py — Production batch driver for the RAG semantic Moving
Targets pipeline (port of NB04 ``04_rag_matching_v4.ipynb``, Section 6 +
Cell 40).

What this does
--------------
1. Load LLM (or spaCy) targets from ``data/processed/`` and unify them
   into the ``{company_id, quarter, metric_name, context, is_financial}``
   schema NB04 uses.
2. Build (or attach to) a persistent ChromaDB collection at
   ``data/cache/chromadb_experiment``.
3. Index every (company, quarter) target set into the collection so
   future quarters can retrieve their own t-4 history.
4. Iterate every (company, quarter) and call
   ``SemanticContinuityMatcher.compute_batch_semantic_mt`` with
   ``historical_lag=4`` (paper-strict t-4) to produce per-quarter
   semantic MT scores.
5. Write three artifacts:
   - ``data/processed/semantic_mt_scores.parquet`` (one row per CQ)
   - ``data/processed/per_pair_sims.parquet`` (long-form pair log,
     optional, see ``--no-per-pair``)
   - ``data/processed/semantic_mt_scores.meta.json`` (schema sidecar
     covering thresholds / calibration source / retrieval mode /
     timing).

Optionally loads calibrated thresholds from
``data/processed/mt_calibration_result.json`` (produced by
``scripts/run_threshold_calibration.py``). Falls back to the locked
``{maintained: 0.65, rephrased: 0.40}`` values from the EarningsLens v1
release.

Usage
-----
    python scripts/run_rag_matching.py
    python scripts/run_rag_matching.py --source spacy --limit 50
    python scripts/run_rag_matching.py --calibration data/processed/mt_calibration_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("scripts.run_rag_matching")


# Locked v1 release thresholds — used when --calibration is not provided.
LOCKED_THRESHOLDS: Dict[str, float] = {"maintained": 0.65, "rephrased": 0.40}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rag_matching",
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
        "--source",
        choices=("auto", "llm", "spacy"),
        default="auto",
        help=(
            "Which targets parquet to score against. 'auto' prefers "
            "llm_targets.parquet and falls back to spacy_targets.parquet."
        ),
    )
    p.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help=(
            "Persistent ChromaDB directory. Defaults to "
            "<data-dir>/cache/chromadb_experiment."
        ),
    )
    p.add_argument(
        "--collection",
        default="targets_experiment",
        help="ChromaDB collection name (default: targets_experiment).",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model identifier.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Embedding device ('cpu' or 'cuda'; default: cpu).",
    )
    p.add_argument(
        "--historical-lag",
        type=int,
        default=4,
        help="Paper-strict retrieval lag, defaults to 4 (Cohen-Malloy-Nguyen).",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Optional path to mt_calibration_result.json. When supplied, "
            "the calibrated F1-optimal thresholds replace the locked defaults."
        ),
    )
    p.add_argument(
        "--maintained-threshold",
        type=float,
        default=None,
        help="Manual override for the maintained cosine threshold.",
    )
    p.add_argument(
        "--rephrased-threshold",
        type=float,
        default=None,
        help="Manual override for the rephrased (tracked-vs-dropped) threshold.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of (company, quarter) pairs scored (default: 0 = all).",
    )
    p.add_argument(
        "--no-per-pair",
        action="store_true",
        help="Skip writing per_pair_sims.parquet (useful for quick smoke runs).",
    )
    p.add_argument(
        "--skip-index",
        action="store_true",
        help=(
            "Skip the full-index pre-pass. Use only when the ChromaDB "
            "collection has already been built (e.g. by a previous run)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _resolve_thresholds(
    args: argparse.Namespace,
    data_processed: Path,
) -> Tuple[Dict[str, float], List[str]]:
    """Pick the operational thresholds following NB04 Cell 34's preference order.

    Resolution order (matches the notebook):

    1. ``--calibration <path>`` if supplied.
    2. Auto-discover ``data/processed/mt_calibration_result.json`` if it
       exists (NB04 Cell 34 reads this without an explicit flag).
    3. Fall back to the locked v1 thresholds ``{0.65, 0.40}``.
    4. Apply any ``--maintained-threshold`` / ``--rephrased-threshold``
       overrides on top of the resolved baseline.
    """
    notes: List[str] = []
    default_calib_path = data_processed / "mt_calibration_result.json"

    calibration_path: Optional[Path] = None
    if args.calibration:
        calibration_path = args.calibration
    elif default_calib_path.exists():
        calibration_path = default_calib_path

    if calibration_path:
        from src.rag.calibration import load_calibrated_thresholds
        thresholds = load_calibrated_thresholds(calibration_path)
        notes.append(f"calibrated from {calibration_path}")
    else:
        thresholds = dict(LOCKED_THRESHOLDS)
        notes.append("locked v1 fallback {0.65, 0.40}")

    if args.maintained_threshold is not None:
        thresholds["maintained"] = float(args.maintained_threshold)
        notes.append(f"--maintained-threshold override = {args.maintained_threshold}")
    if args.rephrased_threshold is not None:
        thresholds["rephrased"] = float(args.rephrased_threshold)
        notes.append(f"--rephrased-threshold override = {args.rephrased_threshold}")

    # NB04 Cell 34 sanity assert: 0 < rephrased < maintained < 1.
    m, r = thresholds["maintained"], thresholds["rephrased"]
    if not (0.0 < r < m < 1.0):
        logger.warning(
            "Thresholds out of expected range (0 < rephrased < maintained < 1): "
            "maintained=%.4f, rephrased=%.4f. Continuing but downstream classification "
            "may behave unexpectedly.", m, r,
        )

    return thresholds, notes


def _load_targets(data_processed: Path, source: str) -> Tuple["pd.DataFrame", str]:
    """Load and unify a targets parquet into NB04's working schema."""
    import pandas as pd  # lazy

    llm_path = data_processed / "llm_targets.parquet"
    spacy_path = data_processed / "spacy_targets.parquet"

    if source == "auto":
        if llm_path.exists():
            source = "llm"
        elif spacy_path.exists():
            source = "spacy"
        else:
            raise FileNotFoundError(
                f"Neither {llm_path} nor {spacy_path} exists. Run "
                "scripts/run_llm_extraction.py or scripts/run_spacy_baseline.py first."
            )

    if source == "llm":
        if not llm_path.exists():
            raise FileNotFoundError(f"LLM targets parquet missing: {llm_path}")
        raw = pd.read_parquet(llm_path)
        df = pd.DataFrame({
            "company_id":  raw["company_id"].astype(str),
            "quarter":     raw["quarter"].astype(str),
            "metric_name": raw["metric_name"].astype(str),
            "context":     raw.get("raw_text", "").astype(str),
            "is_financial": raw.get("is_financial", True),
        })
    else:  # spacy
        if not spacy_path.exists():
            raise FileNotFoundError(f"spaCy targets parquet missing: {spacy_path}")
        raw = pd.read_parquet(spacy_path)
        quarter = (
            raw["fiscalyear"].astype("Int64").astype(str)
            + "Q"
            + raw["fiscalquarter"].astype("Int64").astype(str)
        )
        df = pd.DataFrame({
            "company_id":   raw["companyid"].astype(str),
            "quarter":      quarter,
            "metric_name":  raw["target_text"].astype(str),
            "context":      raw.get("sentence", "").astype(str),
            "is_financial": raw.get("is_financial", False).astype(bool),
        })

    df = df.dropna(subset=["company_id", "quarter", "metric_name"]).copy()
    df["metric_name"] = df["metric_name"].str.strip()
    df = df[df["metric_name"] != ""].reset_index(drop=True)
    df["company_id"] = df["company_id"].str.replace(r"\.0$", "", regex=True)

    logger.info("Loaded %d %s targets from %s", len(df), source.upper(), data_processed)
    return df, source


def _build_batch_iteration_list(
    targets_df: "pd.DataFrame",
) -> List[Dict[str, Any]]:
    """Replicate NB04 Cell 40's duplicate-aware batch_cqs construction."""
    from collections import defaultdict, Counter
    import pandas as pd  # noqa: F401

    nested: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    dup_counter: Counter = Counter()

    for (company, quarter), grp in targets_df.groupby(["company_id", "quarter"]):
        seen = defaultdict(int)
        target_dicts: List[Dict[str, Any]] = []
        for _, row in grp.iterrows():
            raw_name = str(row["metric_name"]).strip()
            seen[raw_name] += 1
            metric_name = raw_name if seen[raw_name] == 1 else f"{raw_name} (#{seen[raw_name]})"
            if seen[raw_name] > 1:
                dup_counter[(company, quarter, raw_name)] += 1
            td: Dict[str, Any] = {"metric_name": metric_name}
            if "context" in row and row["context"]:
                td["context"] = str(row["context"])
            if "is_financial" in row:
                td["is_financial"] = bool(row["is_financial"])
            target_dicts.append(td)
        nested.setdefault(company, {})[quarter] = target_dicts

    if dup_counter:
        logger.info(
            "Deduped %d duplicate metric_names across %d (company, quarter) groups.",
            sum(dup_counter.values()),
            len(dup_counter),
        )

    batch = [
        {"company_id": c, "quarter": q, "current_targets": td}
        for c, qs in nested.items()
        for q, td in qs.items()
    ]
    return batch


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_dir = args.data_dir
    data_processed = data_dir / "processed"
    chroma_dir = args.chroma_dir or (data_dir / "cache" / "chromadb_experiment")

    out_agg   = data_processed / "semantic_mt_scores.parquet"
    out_pairs = data_processed / "per_pair_sims.parquet"
    out_meta  = data_processed / "semantic_mt_scores.meta.json"

    # Mirror NB04 Cell 40: wipe stale aggregate / per-pair parquet so a
    # partial previous run never bleeds into the new artifacts.
    for stale in (out_agg, out_pairs):
        if stale.exists():
            stale.unlink()
            logger.info("Removed stale %s", stale.name)

    thresholds, threshold_notes = _resolve_thresholds(args, data_processed)
    logger.info("Thresholds: %s (%s)", thresholds, "; ".join(threshold_notes))

    targets_df, source_label = _load_targets(data_processed, args.source)

    batch_cqs = _build_batch_iteration_list(targets_df)
    if args.limit > 0:
        batch_cqs = batch_cqs[: args.limit]
        logger.info("Limit applied: scoring first %d company-quarters.", len(batch_cqs))
    logger.info("Prepared %d company-quarters for scoring.", len(batch_cqs))

    # Lazy imports — these pull in chromadb / sentence-transformers, which we
    # do not want at module import time (e.g. for --help).
    from src.rag.vector_store import TargetVectorStore
    from src.rag.semantic_matcher import SemanticContinuityMatcher

    chroma_dir.mkdir(parents=True, exist_ok=True)
    store = TargetVectorStore(
        persist_dir=str(chroma_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        device=args.device,
    )
    logger.info(
        "ChromaDB ready at %s (collection=%s, count=%d)",
        chroma_dir, args.collection, store.count(),
    )

    if not args.skip_index:
        # Pre-index the full corpus once. compute_batch_semantic_mt then runs
        # with index_current=False so it never re-indexes, which is much faster
        # on large batches (NB04 Section 5b convention).
        n_indexed = 0
        for cq in batch_cqs:
            n_indexed += store.index_targets(
                cq["company_id"], cq["quarter"], cq["current_targets"]
            )
        logger.info("Indexed %d documents.", n_indexed)

    # Note: matcher constructor uses ``vector_store`` and ``default_thresholds``
    # (see SemanticContinuityMatcher.__init__). Pass them by name to avoid the
    # ``store=...`` / ``thresholds=...`` mistake that NB04 Cells 33/40 hide
    # behind their inline variable names.
    matcher = SemanticContinuityMatcher(
        vector_store=store,
        default_thresholds=thresholds,
    )

    t0 = time.perf_counter()
    if args.no_per_pair:
        mt_df = matcher.compute_batch_semantic_mt(
            batch_cqs,
            threshold_config=thresholds,
            historical_lag=args.historical_lag,
            index_current=False,
        )
        per_pair_df = None
    else:
        mt_df, per_pair_df = matcher.compute_batch_semantic_mt(
            batch_cqs,
            threshold_config=thresholds,
            historical_lag=args.historical_lag,
            index_current=False,
            return_per_pair=True,
        )
    elapsed = time.perf_counter() - t0
    logger.info("Computed MT for %d company-quarters in %.2fs.", len(mt_df), elapsed)

    out_agg.parent.mkdir(parents=True, exist_ok=True)
    mt_df.to_parquet(out_agg, index=False, compression="zstd")
    logger.info("Wrote %s (%d rows)", out_agg, len(mt_df))

    if per_pair_df is not None:
        per_pair_df.to_parquet(out_pairs, index=False, compression="zstd")
        logger.info("Wrote %s (%d pair rows)", out_pairs, len(per_pair_df))

    import pandas as pd  # for utc timestamp
    meta = {
        "schema_version": 2,
        "run_at_utc": pd.Timestamp.utcnow().isoformat(),
        "thresholds": thresholds,
        "thresholds_source": "; ".join(threshold_notes),
        "calibrated": args.calibration is not None,
        "retrieval_mode": (
            f"paper-strict t-{args.historical_lag} "
            "(historical_lag set, single quarter)"
        ),
        "source_targets": source_label,
        "n_quarters": int(len(mt_df)),
        "elapsed_seconds": round(elapsed, 1),
        "per_pair_log": (
            None if per_pair_df is None
            else str(out_pairs.relative_to(data_dir.parent))
            if data_dir.parent in out_pairs.parents
            else str(out_pairs)
        ),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", out_meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
