"""
data_manager.py — Data readiness, uploads, and overview dashboard for Target Tracker.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class DataFileSpec:
    key: str
    title: str
    description: str
    candidates: Tuple[Path, ...]
    required: bool = True

    def resolve(self, data_dir: Path) -> Optional[Path]:
        for rel in self.candidates:
            path = data_dir / rel
            if path.exists() and path.is_file():
                return path
        return None


def data_specs(data_dir: Path) -> List[DataFileSpec]:
    return [
        DataFileSpec(
            key="transcripts",
            title="Commitment corpus",
            description="Earnings call transcripts (CIQ or normalized parquet)",
            candidates=(
                Path("raw/ciq_transcripts.parquet"),
                Path("raw/transcripts.parquet"),
            ),
            required=True,
        ),
        DataFileSpec(
            key="calibration",
            title="Calibration labels",
            description="Human-labeled pairs for semantic threshold tuning",
            candidates=(Path("processed/mt_calibration_sample_labeled.csv"),),
            required=False,
        ),
        DataFileSpec(
            key="pipeline_cache",
            title="Analysis cache",
            description="Entity-quarter results from the cache pipeline stage",
            candidates=(Path("cache/demo/pipeline_cache.json"),),
            required=False,
        ),
    ]


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def _transcripts_ready(data_dir: Path) -> bool:
    return any(
        (data_dir / p).exists()
        for p in ("raw/ciq_transcripts.parquet", "raw/transcripts.parquet")
    )


def _workflow_ready(data_dir: Path) -> bool:
    return (data_dir / "processed/llm_targets.parquet").exists()


def render_step_header(data_dir: Path, cache_ready: bool) -> str:
    transcripts_ok = _transcripts_ready(data_dir)
    workflow_ok = _workflow_ready(data_dir)
    results_ok = cache_ready

    def pill(label: str, state: str) -> str:
        return f'<span class="step-pill {state}">{label}</span>'

    data_state = "done" if transcripts_ok else "active"
    wf_state = "done" if workflow_ok else ("active" if transcripts_ok else "pending")
    res_state = "done" if results_ok else ("active" if workflow_ok else "pending")

    return f"""
<div class="app-hero">
  <div class="brand-row">
    <div>
      <h1>Target Tracker</h1>
      <p class="tagline">
        Institutional commitment continuity — extract forward guidance, track what
        changed quarter-over-quarter, and surface dropped-target risk.
      </p>
    </div>
    <span class="corpus-badge">Earnings corpus</span>
  </div>
  <div class="step-row">
    {pill("1 · Data", data_state)}
    {pill("2 · Pipeline", wf_state)}
    {pill("3 · Insights", res_state)}
  </div>
</div>
"""


def render_kpi_row(stats: Dict[str, Any]) -> str:
    def card(label: str, value: str, sub: str = "", cls: str = "") -> str:
        extra = f" {cls}" if cls else ""
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return (
            f'<div class="kpi-card{extra}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{sub_html}</div>'
        )

    cache_cls = "ok" if stats.get("cache_pairs", 0) > 0 else "warn"
    data_cls = "ok" if stats.get("data_ready") else "warn"

    return (
        '<div class="kpi-grid">'
        + card("Entities", str(stats.get("n_tickers", 0)), "in analysis cache", cache_cls)
        + card("Quarters", str(stats.get("n_quarters", 0)), "with results", cache_cls)
        + card("Cached pairs", str(stats.get("cache_pairs", 0)), "entity × quarter", cache_cls)
        + card("Data corpus", "Ready" if stats.get("data_ready") else "Missing", "transcript parquet", data_cls)
        + card("Pipeline", stats.get("pipeline_stage", "—"), stats.get("pipeline_detail", ""))
        + "</div>"
    )


def _flag_class(flag: str) -> str:
    f = (flag or "").upper()
    if f == "HIGH":
        return "flag-high"
    if f == "MEDIUM":
        return "flag-medium"
    return "flag-low"


def render_watchlist_preview(rows: List[Dict[str, Any]], quarter: str) -> str:
    if not rows:
        return (
            '<div class="empty-state">No watchlist data yet. Run the <strong>cache</strong> '
            "pipeline stage to populate entity risk scores.</div>"
        )

    body = []
    for r in rows[:8]:
        flag = str(r.get("risk_flag", "LOW"))
        body.append(
            f"<tr>"
            f"<td><strong>{r.get('ticker', '')}</strong></td>"
            f"<td>{float(r.get('mt_score', r.get('mt_score_llm', r.get('risk_score', 0))) or 0):.3f}</td>"
            f"<td>{int(r.get('n_dropped', 0) or 0)}</td>"
            f'<td class="{_flag_class(flag)}">{flag}</td>'
            f"</tr>"
        )

    return f"""
<div class="panel">
  <div class="panel-title">Top risk — {quarter}</div>
  <table class="mini-table">
    <thead><tr>
      <th>Entity</th><th>MT score</th><th>Dropped</th><th>Flag</th>
    </tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
</div>
"""


def render_overview_dashboard(
    data_dir: Path,
    stats: Dict[str, Any],
    watchlist_rows: List[Dict[str, Any]],
    watchlist_quarter: str,
) -> str:
    quickstart = """
<ol class="quickstart-list">
  <li>Download transcript data and upload it on the <strong>Data</strong> tab.</li>
  <li>Run pipeline stages on <strong>Pipeline</strong> (paste your LLM API key first).</li>
  <li>Review per-entity reports on <strong>Entity Report</strong> or scan the <strong>Watchlist</strong>.</li>
</ol>
"""

    watchlist_html = render_watchlist_preview(watchlist_rows, watchlist_quarter or "—")

    status_msg = stats.get("status_message", "")
    status_panel = ""
    if status_msg:
        status_panel = (
            f'<div class="panel"><div class="panel-title">System status</div>'
            f'<p class="panel-subtitle">{status_msg}</p></div>'
        )

    return (
        render_kpi_row(stats)
        + status_panel
        + '<div class="two-col">'
        + '<div class="panel"><div class="panel-title">Quick start</div>'
        + quickstart
        + "</div>"
        + watchlist_html
        + "</div>"
    )


def render_data_dashboard(data_dir: Path) -> str:
    cards: List[str] = []
    for spec in data_specs(data_dir):
        found = spec.resolve(data_dir)
        if found:
            stat = found.stat()
            cls = "ready"
            badge = '<span class="badge ready">Ready</span>'
            meta = f"{found.name} · {_fmt_size(stat.st_size)}"
        elif spec.required:
            cls = "missing"
            badge = '<span class="badge missing">Required</span>'
            meta = "Not found — upload or place in data/"
        else:
            cls = "optional-missing"
            badge = '<span class="badge optional">Optional</span>'
            meta = "Not present yet"

        req = "Required" if spec.required else "Optional"
        cards.append(
            f"""
<div class="file-card {cls}">
  <div style="display:flex;justify-content:space-between;gap:8px;align-items:start;">
    <h4>{spec.title}</h4>
    {badge}
  </div>
  <p class="meta">{spec.description}</p>
  <p class="meta"><strong>{req}</strong> · {meta}</p>
</div>
"""
        )

    raw_scan = scan_raw_universe(data_dir)
    scan_line = ""
    if raw_scan:
        tickers, quarters = raw_scan
        scan_line = (
            f"<p class='meta' style='margin-top:12px;'>Detected in corpus: "
            f"<strong>{len(tickers)}</strong> entities, "
            f"<strong>{len(quarters)}</strong> quarters.</p>"
        )

    return (
        '<div class="panel"><div class="panel-title">Corpus files</div>'
        f'<div class="file-grid">{"".join(cards)}</div>{scan_line}</div>'
    )


def scan_raw_universe(data_dir: Path) -> Optional[Tuple[List[str], List[str]]]:
    """Return tickers/quarters found in the raw transcript parquet, if any."""
    for rel in ("raw/ciq_transcripts.parquet", "raw/transcripts.parquet"):
        path = data_dir / rel
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=None)
        except Exception:
            continue

        tickers: set = set()
        if "ticker" in df.columns:
            tickers |= set(df["ticker"].astype(str).str.upper().dropna().unique())
        quarters: set = set()
        if "quarter" in df.columns:
            quarters |= set(df["quarter"].astype(str).dropna().unique())
        elif {"fiscalyear", "fiscalquarter"}.issubset(df.columns):
            q = (
                df["fiscalyear"].astype("Int64").astype(str)
                + "Q"
                + df["fiscalquarter"].astype("Int64").astype(str)
            )
            quarters |= set(q.dropna().unique())

        if tickers or quarters:
            return sorted(tickers), sorted(quarters)
    return None


def _target_path_for_upload(filename: str, data_dir: Path) -> Path:
    name = Path(filename).name.lower()
    if name.endswith(".csv"):
        return data_dir / "processed" / Path(filename).name
    if name.endswith(".parquet"):
        return data_dir / "raw" / Path(filename).name
    if name.endswith(".json"):
        return data_dir / "cache" / "demo" / Path(filename).name
    return data_dir / "raw" / Path(filename).name


def ingest_uploads(
    files: Optional[Sequence],
    data_dir: Path,
) -> Tuple[str, str]:
    """Save uploaded files into the correct data/ subtree."""
    if not files:
        return render_data_dashboard(data_dir), "No files selected."

    saved: List[str] = []
    for item in files:
        src = getattr(item, "name", None) or str(item)
        src_path = Path(src)
        if not src_path.exists():
            continue
        dest = _target_path_for_upload(src_path.name, data_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        saved.append(str(dest.relative_to(data_dir)))

    if not saved:
        return render_data_dashboard(data_dir), "Upload failed — no files were saved."

    msg = "Saved: " + ", ".join(saved)
    return render_data_dashboard(data_dir), msg
