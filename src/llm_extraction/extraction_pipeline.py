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

    Most fields map 1:1 to ``LLMTargetExtractor.__init__`` parameters and
    expose the knobs that NB03 v2 actually used in production. Anything not
    listed here was hard-coded in the notebook driver cells; if you find
    yourself patching ``run_pipeline`` to pass a new kwarg, add it here so
    the CLI stays consistent.
    """

    data_dir: str = "data"
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    # Default base_url is empty so the SDK uses its own default
    # (api.openai.com for ``backend="openai"``). Pass --base-url to override.
    # The previous default of ``http://localhost:8000/v1`` poisoned every
    # Gemini / OpenAI run because the wiring below always forwarded a non-
    # empty value to ``LLMTargetExtractor``.
    base_url: str = ""

    # Async / rate-limit knobs (NB03 Cell 27 default values)
    max_concurrent: int = 10
    max_completion_tokens: int = 2048
    max_input_tokens_per_chunk: int = 12_000
    request_timeout_s: float = 120.0
    max_retries: int = 6
    rpm_cap: Optional[int] = 600
    tpm_cap: int = 2_000_000

    temperature: float = 0.0

    spacy_baseline_path: str = ""
    output_filename: str = "llm_targets.parquet"
    raw_input_path: str = ""
    processed_output_dir: str = ""
    limit: int = 0

    # Resumable JSONL-first flow (NB03 Cell 42). When True, run
    # ``extract_corpus_to_jsonl`` writing to ``output_filename`` with .jsonl
    # extension first, then flatten to parquet via :func:`save_results_from_jsonl`.
    use_jsonl_flow: bool = False
    jsonl_filename: str = "llm_targets.jsonl"

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
    """Normalize notebook and WRDS CIQ transcript schemas for LLM loading.

    Output schema after normalization:

        - ``transcript_id`` : ``f"{company_id}_{YYYY}Q{N}"`` (NB03 Cell 6
          canonical form). The leading company id keeps any ``.0`` suffix
          pandas added when companyid was stored as a float — downstream
          :func:`parse_transcript_id` strips it via ``split('.')[0]``.
        - ``text``           : component text.
        - ``component_type`` : CIQ component_type_id (2=Presentation,
          3=Analyst Question, 4=Management Answer; 1=Press Release dropped
          to match NB03 Cell 5).
        - ``componentorder`` : preserved when present, for stable sorting.
    """
    import pandas as pd

    df = df.copy()
    rename_map = {
        "transcriptid": "transcript_id_raw",   # do NOT clobber a real transcript_id
        "componenttext": "text",
        "component_type_id": "component_type",
    }
    for source, target in rename_map.items():
        if target not in df.columns and source in df.columns:
            df = df.rename(columns={source: target})

    if "text" not in df.columns:
        raise ValueError("Parquet transcript input must contain text or componenttext")

    # Derive fiscalyear / fiscalquarter when the CIQ retrieval output only
    # has 'quarter' (YYYYQN string) or only an event_date.
    if "fiscalyear" not in df.columns and "year" in df.columns:
        df["fiscalyear"] = df["year"]
    if "fiscalquarter" not in df.columns and "quarter" in df.columns:
        parsed = (
            df["quarter"].astype(str).str.extract(r"(?:^|.*Q)([1-4])$", expand=False)
        )
        df["fiscalquarter"] = pd.to_numeric(parsed, errors="coerce").astype("Int64")
    if (
        ("fiscalyear" not in df.columns or "fiscalquarter" not in df.columns)
        and "event_date" in df.columns
    ):
        ed = pd.to_datetime(df["event_date"], errors="coerce")
        if "fiscalyear" not in df.columns:
            df["fiscalyear"] = ed.dt.year
        if "fiscalquarter" not in df.columns:
            df["fiscalquarter"] = ed.dt.quarter

    # Component-type filter to match NB03 Cell 5 (keep 2 / 3 / 4 only).
    if "component_type" in df.columns:
        ctype_num = pd.to_numeric(df["component_type"], errors="coerce")
        df = df[ctype_num.isin([2, 3, 4])].copy()
    else:
        df["component_type"] = 0

    # Build canonical transcript_id ``{company_id}_{YYYY}Q{N}``. Falls back
    # to the raw ``transcriptid`` integer / a synthesized id only when the
    # canonical fields are missing.
    if (
        "companyid" in df.columns
        and "fiscalyear" in df.columns
        and "fiscalquarter" in df.columns
    ):
        cid = df["companyid"].astype(str)
        fy = df["fiscalyear"].astype("Int64").astype(str)
        fq = df["fiscalquarter"].astype("Int64").astype(str)
        df["transcript_id"] = cid + "_" + fy + "Q" + fq
    elif "transcript_id" in df.columns:
        df["transcript_id"] = df["transcript_id"].astype(str)
    elif "transcript_id_raw" in df.columns:
        df["transcript_id"] = df["transcript_id_raw"].astype(str)
    else:
        df["transcript_id"] = df.index.astype(str)

    # Preserve componentorder if present so groupby keeps call order.
    if "componentorder" in df.columns:
        df = df.sort_values(["transcript_id", "componentorder"])
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
        grouped = df.groupby("transcript_id", sort=False)
        for tid, group in grouped:
            components = group[["text", "component_type"]].to_dict("records")
            # NB03 Cell 6: emit top-level company_id + quarter so
            # extract_corpus_to_jsonl can stamp them on every JSONL line
            # without re-parsing transcript_id later.
            head = group.iloc[0]
            doc: Dict[str, Any] = {"transcript_id": str(tid), "components": components}
            if "companyid" in group.columns:
                doc["company_id"] = str(head["companyid"])
            if "fiscalyear" in group.columns and "fiscalquarter" in group.columns:
                try:
                    fy = int(head["fiscalyear"])
                    fq = int(head["fiscalquarter"])
                    doc["quarter"] = f"{fy}Q{fq}"
                except (TypeError, ValueError):
                    pass
            transcripts.append(doc)
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

import re

# Match NB03's transcript_id format: "{company_id}_{YYYY}Q{N}" or
# "{company_id}.0_{YYYY}Q{N}" (the .0 suffix shows up because pandas casts
# numeric companyids through float at JSONL flatten time).
_TRANSCRIPT_ID_RE = re.compile(
    r"^(?P<company_id>\d+(?:\.\d+)?)_(?P<fiscalyear>\d{4})Q(?P<fiscalquarter>[1-4])$"
)


def parse_transcript_id(transcript_id: str) -> Dict[str, Any]:
    """
    Split an NB03 transcript_id into ``company_id``, ``fiscalyear``,
    ``fiscalquarter``, and ``quarter`` (``YYYYQN``).

    Returns a dict with all four keys, with ``None`` values when parsing
    fails so the caller can still write the row.

    >>> parse_transcript_id("18711.0_2023Q4") == {
    ...     "company_id": "18711", "fiscalyear": 2023,
    ...     "fiscalquarter": 4, "quarter": "2023Q4",
    ... }
    True
    >>> parse_transcript_id("free-form-id")
    {'company_id': 'free-form-id', 'fiscalyear': None, 'fiscalquarter': None, 'quarter': ''}
    """
    if not transcript_id:
        return {"company_id": "", "fiscalyear": None, "fiscalquarter": None, "quarter": ""}

    m = _TRANSCRIPT_ID_RE.match(str(transcript_id))
    if not m:
        return {
            "company_id": str(transcript_id),
            "fiscalyear": None,
            "fiscalquarter": None,
            "quarter": "",
        }

    cid_raw = m.group("company_id")
    # Split on '.' so companyids ending in 0 (like IBM's 112350) keep their
    # trailing zeros — the previous str.rstrip('.0') heuristic had a known
    # truncation bug, see NB06 Cell 5.
    company_id = cid_raw.split(".")[0]
    fiscalyear = int(m.group("fiscalyear"))
    fiscalquarter = int(m.group("fiscalquarter"))
    return {
        "company_id": company_id,
        "fiscalyear": fiscalyear,
        "fiscalquarter": fiscalquarter,
        "quarter": f"{fiscalyear}Q{fiscalquarter}",
    }


def _load_results_from_jsonl(jsonl_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load an ``llm_targets.jsonl`` file into the in-memory results dict."""
    results: Dict[str, List[Dict[str, Any]]] = {}
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning("JSONL output missing at %s", path)
        return results

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line: %s", exc)
                continue
            tid = rec.get("transcript_id")
            if not tid:
                continue
            results[tid] = list(rec.get("targets") or [])
    return results


def save_results(
    results: Dict[str, List[Dict]],
    output_path: str,
) -> int:
    """
    Flatten extraction results and save to Parquet.

    Each output row carries the original target fields plus the
    ``transcript_id`` and the parsed metadata columns (``company_id``,
    ``fiscalyear``, ``fiscalquarter``, ``quarter``) so downstream scripts
    do not have to re-parse the ID. Numeric ``numerical_value`` is also
    coerced to float — NB03 Cell 43.

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
        meta = parse_transcript_id(transcript_id)
        for target in targets:
            rows.append({
                "transcript_id": transcript_id,
                **meta,
                **target,
            })

    if not rows:
        logger.warning("save_results | no targets to save — writing empty file")
        df = pd.DataFrame(columns=["transcript_id"])
    else:
        df = pd.DataFrame(rows)
        if "fiscalyear" in df.columns:
            df["fiscalyear"] = df["fiscalyear"].astype("Int64")
        if "fiscalquarter" in df.columns:
            df["fiscalquarter"] = df["fiscalquarter"].astype("Int64")
        if "numerical_value" in df.columns:
            df["numerical_value"] = pd.to_numeric(
                df["numerical_value"], errors="coerce"
            )
        # Drop the Unicode-garbage columns that earlier prompt corruption
        # leaked in (NB06 Cell 5 + Cell 12 both strip these). Keeps
        # llm_targets.parquet schema-clean even when the JSONL has noise.
        garbage = [c for c in df.columns
                   if isinstance(c, str)
                   and c.startswith("temporal_")
                   and c != "temporal_framing"]
        if garbage:
            df = df.drop(columns=garbage)
            logger.info("save_results | dropped %d garbage temporal_* columns: %s",
                        len(garbage), garbage)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("save_results | saved %d rows to %s", len(df), output_path)
    return len(df)


def repair_parquet_from_jsonl(
    jsonl_path: str,
    parquet_path: str,
    *,
    backup_suffix: str = "_buggy_company_id",
) -> Dict[str, int]:
    """
    Rebuild ``llm_targets.parquet`` from the canonical JSONL.

    Mirrors NB06 Cell 5: the original NB03 parquet writer used
    ``str.rstrip('.0')``, which silently truncated company IDs ending in
    zero (e.g. IBM ``112350`` → ``11235``). This helper reads the JSONL
    (source of truth), parses ``transcript_id`` correctly via
    :func:`parse_transcript_id`, and overwrites the parquet — keeping a
    backup of the previous file so nothing is lost.

    Returns
    -------
    Dict[str, int]
        ``{"rows": N, "transcripts": K, "companies_old": .., "companies_new": ..}``
    """
    try:
        import pandas as pd  # noqa: F401  (only used through save_results)
    except ImportError as exc:
        raise ImportError("pandas required: pip install pandas pyarrow") from exc

    jsonl_p = Path(jsonl_path)
    parquet_p = Path(parquet_path)
    if not jsonl_p.exists():
        raise FileNotFoundError(
            f"JSONL source-of-truth missing: {jsonl_p}. "
            "Cannot repair parquet without the raw rows."
        )

    if parquet_p.exists():
        backup = parquet_p.with_name(parquet_p.stem + backup_suffix + parquet_p.suffix)
        if not backup.exists():
            import shutil
            shutil.copy2(parquet_p, backup)
            logger.info("Backed up buggy parquet -> %s", backup)

    results = _load_results_from_jsonl(str(jsonl_p))
    n_rows = save_results(results, str(parquet_p))
    return {
        "rows": n_rows,
        "transcripts": len(results),
    }


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
    # NB03 v2 knobs are exposed 1:1 as kwargs. The previous version of
    # this code passed ``max_tokens_per_request``, which is NOT a parameter
    # accepted by ``LLMTargetExtractor.__init__`` — the value was silently
    # dropped and the per-request token budget always defaulted. Bug fixed.
    extractor_kwargs: Dict[str, Any] = {
        "backend": config.backend,
        "model": config.model,
        "max_concurrent": config.max_concurrent,
        "max_completion_tokens": config.max_completion_tokens,
        "max_input_tokens_per_chunk": config.max_input_tokens_per_chunk,
        "request_timeout_s": config.request_timeout_s,
        "max_retries": config.max_retries,
        "rpm_cap": config.rpm_cap,
        "tpm_cap": config.tpm_cap,
        "temperature": config.temperature,
    }
    if config.api_key:
        extractor_kwargs["api_key"] = config.api_key
    # Forward base_url ONLY when explicitly set, so the SDK default
    # (api.openai.com for openai backend, server config for local) wins
    # otherwise. Local-only backends still need a base_url; require the
    # caller to pass --base-url for them.
    if config.base_url:
        extractor_kwargs["base_url"] = config.base_url

    logger.info("run_pipeline | initialising LLMTargetExtractor (backend=%s) …", config.backend)
    extractor = LLMTargetExtractor(**extractor_kwargs)

    # ── Resolve output paths ───────────────────────────────────────────────
    processed_dir = (
        Path(config.processed_output_dir)
        if config.processed_output_dir
        else Path(config.data_dir) / "processed"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(processed_dir / config.output_filename)
    jsonl_path = str(processed_dir / config.jsonl_filename)

    # ── Run async extraction ───────────────────────────────────────────────
    logger.info(
        "run_pipeline | starting extraction | %d transcripts | max_concurrent=%d | jsonl_flow=%s",
        len(transcripts), config.max_concurrent, config.use_jsonl_flow,
    )
    t_extract = time.monotonic()

    if config.use_jsonl_flow:
        # Resumable production path (NB03 Cell 42). Writes one line per
        # transcript as it completes, so a Ctrl-C / 503 mid-run does not
        # lose work — the next launch skips finished transcript_ids.
        # NOTE: extract_corpus_to_jsonl accepts the parameter as ``out_path``;
        # passing ``output_path=`` raises TypeError (the audit flagged this).
        await extractor.extract_corpus_to_jsonl(
            transcripts,
            out_path=jsonl_path,
            max_concurrent=config.max_concurrent,
        )
        results: Dict[str, List[Dict[str, Any]]] = _load_results_from_jsonl(jsonl_path)
    else:
        # In-memory path (suitable for short interactive runs).
        results = await extractor.extract_corpus(
            transcripts, max_concurrent=config.max_concurrent
        )

    elapsed_extract = time.monotonic() - t_extract
    logger.info(
        "run_pipeline | extraction complete in %.2fs | %d transcripts processed",
        elapsed_extract, len(results),
    )

    # ── Save results ───────────────────────────────────────────────────────
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
        default="",
        help=(
            "Override the SDK base URL. Required for local vLLM/Ollama (e.g. "
            "http://localhost:8000/v1) and useful for Gemini's OpenAI-compatible "
            "endpoint (https://generativelanguage.googleapis.com/v1beta/openai/). "
            "Default empty = use the SDK's own default (api.openai.com)."
        ),
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
        "--max-completion-tokens",
        dest="max_completion_tokens",
        type=int,
        default=2048,
        help="Per-request completion token budget (default: 2048)",
    )
    p.add_argument(
        "--max-input-tokens-per-chunk",
        type=int,
        default=12_000,
        help="Cap on the input-token count per LLM chunk (default: 12000, NB03 v2 setting)",
    )
    p.add_argument(
        "--request-timeout-s",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120, NB03 v2 setting)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Async retry budget for transient errors (default: 6)",
    )
    p.add_argument(
        "--rpm-cap",
        type=int,
        default=600,
        help="Requests-per-minute cap (default: 600, NB03 v2 setting)",
    )
    p.add_argument(
        "--tpm-cap",
        type=int,
        default=2_000_000,
        help="Tokens-per-minute cap (default: 2_000_000)",
    )
    p.add_argument(
        "--use-jsonl-flow",
        action="store_true",
        help=(
            "Use the resumable JSONL-first flow (NB03 Cell 42). "
            "Each transcript's targets are appended to llm_targets.jsonl as "
            "they finish, so a Ctrl-C / 503 mid-run does not lose work."
        ),
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
        max_completion_tokens=args.max_completion_tokens,
        max_input_tokens_per_chunk=args.max_input_tokens_per_chunk,
        request_timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        rpm_cap=args.rpm_cap,
        tpm_cap=args.tpm_cap,
        temperature=args.temperature,
        spacy_baseline_path=args.spacy_baseline,
        output_filename=args.output,
        raw_input_path=args.raw_input_path,
        processed_output_dir=args.output_dir,
        limit=args.limit,
        use_jsonl_flow=args.use_jsonl_flow,
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
