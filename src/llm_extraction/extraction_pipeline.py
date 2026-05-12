"""
extraction_pipeline.py — End-to-end LLM extraction pipeline for EarningsLens.

Orchestrates the full workflow:
  1. Load transcripts from data/raw/
  2. Run LLM extraction (async) across all transcripts
  3. Save results to data/processed/llm_targets.parquet
  4. Compare target counts against spaCy baseline
  5. Log and save summary statistics

Intended to be run as a script or imported and driven programmatically:

    # CLI usage
    python -m earningslens.src.llm_extraction.extraction_pipeline \
        --backend openai \
        --model gpt-4o-mini \
        --data-dir /path/to/earningslens/data \
        --max-concurrent 10

    # Programmatic usage
    from earningslens.src.llm_extraction.extraction_pipeline import run_pipeline
    asyncio.run(run_pipeline(config))
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration for the LLM extraction pipeline.

    Attributes
    ----------
    data_dir       : str  — root data directory (expects data_dir/raw/ and
                            data_dir/processed/ sub-dirs)
    backend        : str  — "openai" or "local"
    model          : str  — model identifier for chosen backend
    api_key        : str  — OpenAI API key (overrides env var)
    base_url       : str  — base URL for local vLLM/Ollama server
    max_concurrent : int  — max simultaneous LLM requests
    max_tokens     : int  — per-request token budget
    temperature    : float— sampling temperature (0 = deterministic)
    spacy_baseline_path : str — path to spaCy baseline parquet for comparison
    output_filename     : str — output parquet filename under data_dir/processed/
    log_level      : str  — Python logging level name
    """
    data_dir: str = "data"
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "http://localhost:8000/v1"
    max_concurrent: int = 10
    max_tokens: int = 2048
    temperature: float = 0.0
    spacy_baseline_path: str = ""
    output_filename: str = "llm_targets.parquet"
    raw_input_path: str = ""
    processed_output_dir: str = ""
    limit: int = 0
    log_level: str = "INFO"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transcript loader
# ---------------------------------------------------------------------------

def _find_transcript_parquet(raw_path: Path) -> Optional[Path]:
    """Return the transcript parquet file in *raw_path*, if present."""
    for filename in ("transcripts.parquet", "ciq_transcripts.parquet"):
        candidate = raw_path / filename
        if candidate.exists():
            return candidate
    return None


def _normalize_transcript_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize notebook and WRDS CIQ transcript schemas for LLM loading."""
    df = df.copy()
    rename_map = {
        "transcriptid": "transcript_id",
        "componenttext": "text",
        "component_type_id": "component_type",
    }
    for source, target in rename_map.items():
        if target not in df.columns and source in df.columns:
            df = df.rename(columns={source: target})

    if "transcript_id" not in df.columns:
        group_cols = [
            c for c in ["companyid", "fiscalyear", "fiscalquarter", "year", "quarter"]
            if c in df.columns
        ]
        if group_cols:
            df["transcript_id"] = df[group_cols].astype(str).agg("_".join, axis=1)
        else:
            df["transcript_id"] = df.index.astype(str)
    if "component_type" not in df.columns:
        df["component_type"] = 0
    if "text" not in df.columns:
        raise ValueError("Parquet transcript input must contain text or componenttext")
    return df


def load_transcripts(raw_dir: str) -> List[Dict]:
    """
    Load all transcript documents from *raw_dir*.

    ``raw_dir`` may be either a directory or a direct ``.parquet`` file path.
    Supports two formats:
      • Individual ``.json`` files — one transcript per file. Each JSON file
        must be an object with a ``"transcript_id"`` key and a ``"components"``
        list. If no ``"transcript_id"`` is present the filename stem is used.
      • A single ``transcripts.parquet`` / ``ciq_transcripts.parquet`` file
        (or explicit parquet input) with columns such as ``transcript_id`` or
        ``transcriptid``, ``component_type`` or ``component_type_id``, and
        ``text`` or ``componenttext``.

    Parameters
    ----------
    raw_dir : str — directory to scan for transcript files

    Returns
    -------
    List[Dict] — list of transcript documents, each with:
        "transcript_id" : str
        "components"    : List[Dict]  (each has "text", "component_type")
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_dir}")

    transcripts: List[Dict] = []

    # ── Parquet path ───────────────────────────────────────────────────────
    parquet_file = raw_path if raw_path.is_file() else _find_transcript_parquet(raw_path)
    if parquet_file is not None and parquet_file.exists() and parquet_file.suffix == ".parquet":
        logger.info("load_transcripts | reading %s", parquet_file)
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required: pip install pandas pyarrow") from exc

        df = pd.read_parquet(parquet_file)
        df = _normalize_transcript_dataframe(df)
        grouped = df.groupby("transcript_id")
        for tid, group in grouped:
            components = group[["text", "component_type"]].to_dict("records")
            transcripts.append({"transcript_id": str(tid), "components": components})
        logger.info("load_transcripts | loaded %d transcripts from parquet", len(transcripts))
        return transcripts

    # ── JSON files path ────────────────────────────────────────────────────
    if raw_path.is_file():
        raise FileNotFoundError(f"Unsupported transcript input file: {raw_path}")

    json_files = sorted(raw_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No transcripts.parquet or *.json files found in {raw_dir}"
        )

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if not isinstance(doc, dict):
                logger.warning("load_transcripts | skipping %s (not a JSON object)", json_file)
                continue
            # Normalise: ensure transcript_id and components keys exist
            doc.setdefault("transcript_id", json_file.stem)
            if "components" not in doc:
                # Fallback: treat the whole document text as one component
                text = doc.get("text") or doc.get("content") or ""
                doc["components"] = [{"text": text, "component_type": 0}]
            transcripts.append(doc)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("load_transcripts | could not load %s: %s", json_file, exc)

    logger.info("load_transcripts | loaded %d transcripts from JSON files", len(transcripts))
    return transcripts


# ---------------------------------------------------------------------------
# Results saver
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, List[Dict]],
    output_path: str,
) -> int:
    """
    Flatten extraction results and save to Parquet.

    Parameters
    ----------
    results     : Dict[str, List[Dict]] — {transcript_id: [target_dicts]}
    output_path : str — destination Parquet file path

    Returns
    -------
    int — number of rows saved
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas required: pip install pandas pyarrow") from exc

    rows = []
    for transcript_id, targets in results.items():
        for target in targets:
            row = {"transcript_id": transcript_id, **target}
            rows.append(row)

    if not rows:
        logger.warning("save_results | no targets to save — writing empty file")
        df = pd.DataFrame(columns=["transcript_id"])
    else:
        df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("save_results | saved %d rows to %s", len(df), output_path)
    return len(df)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_with_baseline(
    llm_results: Dict[str, List[Dict]],
    baseline_path: str,
) -> Dict[str, Any]:
    """
    Compare LLM extraction counts against spaCy baseline per transcript.

    Parameters
    ----------
    llm_results   : Dict[str, List[Dict]] — LLM extraction output
    baseline_path : str — path to spaCy targets parquet (must have
                          ``transcript_id`` column)

    Returns
    -------
    Dict with keys:
        "llm_total"       : int
        "spacy_total"     : int
        "llm_only_total"  : int   — targets in LLM but NOT in spaCy
        "delta_pct"       : float — (llm - spacy) / spacy * 100
        "per_transcript"  : List[Dict]
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("compare_with_baseline | pandas not available, skipping comparison")
        return {}

    if not Path(baseline_path).exists():
        logger.warning(
            "compare_with_baseline | baseline not found at %s, skipping", baseline_path
        )
        return {}

    baseline_df = pd.read_parquet(baseline_path)
    spacy_counts: Dict[str, int] = (
        baseline_df.groupby("transcript_id").size().to_dict()
        if "transcript_id" in baseline_df.columns
        else {}
    )

    llm_counts = {tid: len(targets) for tid, targets in llm_results.items()}
    all_tids = set(llm_counts) | set(spacy_counts)

    per_transcript = []
    for tid in sorted(all_tids):
        llm_n = llm_counts.get(tid, 0)
        spacy_n = spacy_counts.get(tid, 0)
        delta = llm_n - spacy_n
        per_transcript.append({
            "transcript_id": tid,
            "llm_targets": llm_n,
            "spacy_targets": spacy_n,
            "delta": delta,
        })

    llm_total = sum(llm_counts.values())
    spacy_total = sum(spacy_counts.values())
    delta_pct = ((llm_total - spacy_total) / spacy_total * 100) if spacy_total else float("nan")

    summary = {
        "llm_total": llm_total,
        "spacy_total": spacy_total,
        "delta_pct": round(delta_pct, 2),
        "per_transcript": per_transcript,
    }

    logger.info(
        "compare_with_baseline | LLM=%d  spaCy=%d  Δ=%.1f%%",
        llm_total, spacy_total, delta_pct,
    )
    return summary


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Compute summary statistics over the extracted targets.

    Parameters
    ----------
    results : Dict[str, List[Dict]] — extraction results

    Returns
    -------
    Dict with aggregate counts, confidence distribution, financial breakdown, etc.
    """
    all_targets = [t for targets in results.values() for t in targets]
    n = len(all_targets)

    if n == 0:
        return {"total_targets": 0}

    confidence_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    temporal_counts: Dict[str, int] = {}
    financial_count = 0

    for t in all_targets:
        conf = t.get("confidence", "low")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        temporal = t.get("temporal_framing", "unknown")
        temporal_counts[temporal] = temporal_counts.get(temporal, 0) + 1
        if t.get("is_financial", False):
            financial_count += 1

    targets_per_transcript = {
        "min": min(len(v) for v in results.values()),
        "max": max(len(v) for v in results.values()),
        "mean": round(n / len(results), 2) if results else 0,
    }

    return {
        "total_targets": n,
        "total_transcripts": len(results),
        "targets_per_transcript": targets_per_transcript,
        "confidence_distribution": confidence_counts,
        "temporal_framing_distribution": temporal_counts,
        "financial_targets": financial_count,
        "operational_targets": n - financial_count,
    }


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    Execute the full LLM extraction pipeline.

    Parameters
    ----------
    config : PipelineConfig — pipeline settings

    Returns
    -------
    Dict — final summary with keys:
        "results_path", "total_targets", "comparison", "stats", "telemetry"
    """
    from .llm_extractor import LLMTargetExtractor

    # ── Configure logging ──────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("run_pipeline | config: %s", asdict(config))

    # ── Load transcripts ───────────────────────────────────────────────────
    raw_dir = config.raw_input_path or str(Path(config.data_dir) / "raw")
    logger.info("run_pipeline | loading transcripts from %s …", raw_dir)
    t0 = time.monotonic()
    transcripts = load_transcripts(raw_dir)
    logger.info(
        "run_pipeline | loaded %d transcripts in %.2fs",
        len(transcripts), time.monotonic() - t0,
    )

    if config.limit > 0:
        transcripts = transcripts[: config.limit]
        logger.info("run_pipeline | limited run to first %d transcripts", len(transcripts))

    if not transcripts:
        logger.error("run_pipeline | no transcripts found — aborting")
        return {}

    # ── Build extractor ────────────────────────────────────────────────────
    extractor_kwargs: Dict[str, Any] = {
        "backend": config.backend,
        "model": config.model,
        "max_tokens_per_request": config.max_tokens,
        "temperature": config.temperature,
    }
    if config.api_key:
        extractor_kwargs["api_key"] = config.api_key
    if config.backend == "local":
        extractor_kwargs["base_url"] = config.base_url

    logger.info("run_pipeline | initialising LLMTargetExtractor (backend=%s) …", config.backend)
    extractor = LLMTargetExtractor(**extractor_kwargs)

    # ── Run async extraction ───────────────────────────────────────────────
    logger.info(
        "run_pipeline | starting extraction | %d transcripts | max_concurrent=%d",
        len(transcripts), config.max_concurrent,
    )
    t_extract = time.monotonic()
    results = await extractor.extract_corpus(
        transcripts, max_concurrent=config.max_concurrent
    )
    elapsed_extract = time.monotonic() - t_extract
    logger.info(
        "run_pipeline | extraction complete in %.2fs | %d transcripts processed",
        elapsed_extract, len(results),
    )

    # ── Save results ───────────────────────────────────────────────────────
    processed_dir = Path(config.processed_output_dir) if config.processed_output_dir else Path(config.data_dir) / "processed"
    output_path = str(processed_dir / config.output_filename)
    n_saved = save_results(results, output_path)

    # ── Compare with spaCy baseline ────────────────────────────────────────
    baseline_path = config.spacy_baseline_path or str(
        processed_dir / "spacy_targets.parquet"
    )
    comparison = compare_with_baseline(results, baseline_path)

    # ── Summary statistics ─────────────────────────────────────────────────
    stats = compute_summary_stats(results)
    telemetry = extractor.telemetry

    summary: Dict[str, Any] = {
        "results_path": output_path,
        "total_targets": n_saved,
        "extraction_time_seconds": round(elapsed_extract, 2),
        "comparison_vs_spacy": comparison,
        "stats": stats,
        "telemetry": telemetry,
    }

    # ── Log summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("run_pipeline | SUMMARY")
    logger.info("  Transcripts processed : %d", len(results))
    logger.info("  Total targets saved   : %d", n_saved)
    logger.info("  Output path           : %s", output_path)
    logger.info("  Extraction time       : %.2fs", elapsed_extract)
    if comparison:
        logger.info(
            "  vs. spaCy baseline    : LLM=%d  spaCy=%d  Δ=%.1f%%",
            comparison.get("llm_total", 0),
            comparison.get("spacy_total", 0),
            comparison.get("delta_pct", 0.0),
        )
    logger.info(
        "  Token usage           : %d total  (%d requests, %d failures)",
        telemetry["total_tokens_used"],
        telemetry["total_requests"],
        telemetry["failed_requests"],
    )
    logger.info("=" * 60)

    # Save summary JSON alongside the parquet
    summary_path = str(processed_dir / "llm_extraction_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("run_pipeline | summary JSON saved to %s", summary_path)
    except OSError as exc:
        logger.warning("run_pipeline | could not save summary JSON: %s", exc)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the pipeline."""
    p = argparse.ArgumentParser(
        prog="extraction_pipeline",
        description="EarningsLens LLM extraction pipeline",
    )
    p.add_argument(
        "--backend",
        choices=["openai", "local"],
        default="openai",
        help="LLM backend to use (default: openai)",
    )
    p.add_argument(
        "--model",
        default="",
        help="Model identifier (default: gpt-4o-mini for openai, Mistral-7B for local)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default: OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL for local vLLM/Ollama server",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing raw/ and processed/ sub-dirs",
    )
    p.add_argument(
        "--input",
        "--raw-input-path",
        dest="raw_input_path",
        default="",
        help="Optional direct transcript input path (parquet file or JSON directory).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of transcripts to process for smoke tests (default: all).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent LLM requests (default: 10)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Per-request token budget (default: 2048)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    p.add_argument(
        "--spacy-baseline",
        default="",
        help="Path to spaCy baseline parquet for comparison",
    )
    p.add_argument(
        "--output",
        default="llm_targets.parquet",
        help="Output filename under data/processed/ (default: llm_targets.parquet)",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Optional processed output directory; overrides data-dir/processed location.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point. Returns exit code (0 = success, 1 = error).

    Parameters
    ----------
    argv : List[str], optional — argument list (defaults to sys.argv[1:])
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        data_dir=args.data_dir,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        spacy_baseline_path=args.spacy_baseline,
        output_filename=args.output,
        raw_input_path=args.raw_input_path,
        processed_output_dir=args.output_dir,
        limit=args.limit,
        log_level=args.log_level,
    )

    try:
        summary = asyncio.run(run_pipeline(config))
        if not summary:
            return 1
        return 0
    except KeyboardInterrupt:
        logger.info("run_pipeline | interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline | fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
