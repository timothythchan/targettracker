"""
status.py — Inspect the EarningsLens data directory and report pipeline state.

The pipeline writes a fixed set of artifacts under ``data/raw/``,
``data/processed/``, and ``data/cache/demo/``. ``earningslens status``
(and the Gradio "Pipeline" tab) calls :func:`describe_pipeline_status`
to render a single readable summary covering all six stages.

This is a read-only helper. It does not modify any files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ArtifactStatus:
    """One pipeline artifact: does it exist on disk, how big, when written."""

    name: str
    path: Path
    exists: bool = False
    size_bytes: int = 0
    mtime: Optional[datetime] = None

    @property
    def size_human(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        if self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_bytes / 1024 / 1024:.1f} MB"
        return f"{self.size_bytes / 1024 ** 3:.2f} GB"

    @property
    def mtime_human(self) -> str:
        if self.mtime is None:
            return "—"
        return self.mtime.strftime("%Y-%m-%d %H:%M")


@dataclass
class StageStatus:
    """One pipeline stage: name, script alias, ready/blocked, artifact list."""

    name: str
    description: str
    cli_subcommand: str
    artifacts: List[ArtifactStatus] = field(default_factory=list)

    @property
    def all_present(self) -> bool:
        return bool(self.artifacts) and all(a.exists for a in self.artifacts)

    @property
    def any_present(self) -> bool:
        return any(a.exists for a in self.artifacts)

    @property
    def state_label(self) -> str:
        if not self.artifacts:
            return "—"
        if self.all_present:
            return "ready"
        if self.any_present:
            return "partial"
        return "missing"


# ---------------------------------------------------------------------------
# Stage definitions — single source of truth for what each subcommand writes
# ---------------------------------------------------------------------------

def _artifact(path: Path, name: Optional[str] = None) -> ArtifactStatus:
    name = name or path.name
    if path.exists() and path.is_file():
        stat = path.stat()
        return ArtifactStatus(
            name=name,
            path=path,
            exists=True,
            size_bytes=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )
    return ArtifactStatus(name=name, path=path, exists=False)


def collect_stage_statuses(data_dir: Path) -> List[StageStatus]:
    """Inspect ``data_dir`` and return one :class:`StageStatus` per pipeline stage."""
    raw = data_dir / "raw"
    processed = data_dir / "processed"
    cache = data_dir / "cache" / "demo"

    return [
        StageStatus(
            name="data",
            description="WRDS data retrieval (NB01)",
            cli_subcommand="earningslens data",
            artifacts=[
                _artifact(raw / "top200_universe.parquet"),
                _artifact(raw / "ciq_transcripts.parquet"),
                _artifact(raw / "crsp_daily.parquet"),
                _artifact(raw / "compustat_fundq.parquet"),
                _artifact(raw / "ibes_statsum.parquet"),
                _artifact(raw / "ff_factors_monthly.parquet"),
            ],
        ),
        StageStatus(
            name="llm",
            description="LLM target extraction (NB03)",
            cli_subcommand="earningslens llm",
            artifacts=[
                _artifact(processed / "llm_targets.parquet"),
                _artifact(processed / "llm_targets.jsonl"),
                _artifact(processed / "llm_extraction_summary.json"),
            ],
        ),
        StageStatus(
            name="rag",
            description="Semantic MT via ChromaDB (NB04)",
            cli_subcommand="earningslens rag",
            artifacts=[
                _artifact(processed / "semantic_mt_scores.parquet"),
                _artifact(processed / "per_pair_sims.parquet"),
                _artifact(processed / "semantic_mt_scores.meta.json"),
            ],
        ),
        StageStatus(
            name="calibrate",
            description="Threshold calibration (NB04b)",
            cli_subcommand="earningslens calibrate",
            artifacts=[
                _artifact(processed / "mt_calibration_sample_labeled.csv"),
                _artifact(processed / "mt_calibration_result.json"),
                _artifact(processed / "semantic_mt_scores_calibrated.meta.json"),
            ],
        ),
        StageStatus(
            name="cache",
            description="Gradio demo cache (NB06)",
            cli_subcommand="earningslens cache",
            artifacts=[
                _artifact(cache / "pipeline_cache.json"),
                _artifact(cache / "portfolio_screen.json"),
                _artifact(cache / "llm_results.json"),
            ],
        ),
    ]


def describe_pipeline_status(data_dir: Path) -> str:
    """Render a human-readable status table for the pipeline."""
    stages = collect_stage_statuses(data_dir)

    lines: List[str] = []
    lines.append(f"EarningsLens pipeline status — data dir: {data_dir}")
    lines.append("=" * 72)
    for stage in stages:
        lines.append(
            f"[{stage.state_label.upper():7}] {stage.name:<10} {stage.description}"
        )
        for art in stage.artifacts:
            mark = "  ok  " if art.exists else "MISSING"
            lines.append(
                f"           {mark}  {art.name:<42} {art.size_human:>10}  {art.mtime_human}"
            )
        lines.append("")
    next_stage = _suggest_next_stage(stages)
    if next_stage is not None:
        lines.append(f"Next stage to run: `{next_stage.cli_subcommand}`")
    else:
        lines.append("All stages have produced artifacts. Run `earningslens app` to launch the UI.")
    return "\n".join(lines)


def _suggest_next_stage(stages: List[StageStatus]) -> Optional[StageStatus]:
    """Return the first stage that has not produced any artifact yet."""
    for stage in stages:
        if not stage.any_present:
            return stage
    return None


def status_dict(data_dir: Path) -> Dict[str, Dict[str, object]]:
    """Machine-readable variant of :func:`describe_pipeline_status`."""
    out: Dict[str, Dict[str, object]] = {}
    for stage in collect_stage_statuses(data_dir):
        out[stage.name] = {
            "description": stage.description,
            "state": stage.state_label,
            "artifacts": [
                {
                    "name": a.name,
                    "path": str(a.path),
                    "exists": a.exists,
                    "size_bytes": a.size_bytes,
                    "mtime": a.mtime.isoformat() if a.mtime else None,
                }
                for a in stage.artifacts
            ],
        }
    return out
