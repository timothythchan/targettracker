"""
demo/app.py — Target Tracker Gradio application handlers.

Launch with the single top-level entry point::

    python app.py

Tabs (assembled in ``demo/interface.py``):

    Overview        — KPI dashboard and quick start
    Data            — corpus upload and file checklist
    Pipeline        — in-browser stage execution
    Entity Report   — per-entity / per-quarter analysis
    Watchlist       — ranked MT-risk screen

Users download data manually, place it under ``data/raw/``, then run stages
from the Pipeline tab. No separate CLI is required.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Silence Gradio 6.0 DeprecationWarning about theme/css moving to launch();
# the kwargs on Blocks(...) still work in Gradio 5.x. Must be set before any
# gradio import or Blocks construction so the filter is registered first.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*'(theme|css)' parameter in the Blocks constructor.*",
)

# Ensure the project root is on the path so src.* imports work
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# LLM config (env-var driven, matches extractor_agent / reporter_agent)
# ---------------------------------------------------------------------------
os.environ.setdefault("EARNINGSLENS_LLM_BACKEND", "openai")
os.environ.setdefault("EARNINGSLENS_LLM_MODEL", "gemini-2.5-flash-lite")
os.environ.setdefault(
    "EARNINGSLENS_LLM_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Surface a unified API key var for any downstream code that reads OPENAI_API_KEY
_LLM_API_KEY = (
    os.getenv("EARNINGSLENS_LLM_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if _LLM_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _LLM_API_KEY

# ---------------------------------------------------------------------------
# Gradio import
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False
    raise ImportError(
        "gradio is required for the demo. Install with: pip install gradio"
    )

# ---------------------------------------------------------------------------
# Pre-computed cache path
# ---------------------------------------------------------------------------
# The cache lives under data/cache/demo and is populated by
# scripts/build_demo_cache.py (the script port of NB06). When the cache is
# empty the app surfaces a "build the cache first" banner instead of
# pretending it has data — synthetic stubs are not an acceptable substitute
# for the real pipeline output.
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "demo"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# True once _discover_cache notices the directory is empty.
_CACHE_IS_EMPTY = False

# ---------------------------------------------------------------------------
# Discover available (ticker, quarter) pairs from cache
# ---------------------------------------------------------------------------
# FIX #1: single-backslash regex inside raw string.
_CACHE_FILE_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<quarter>\d{4}Q[1-4])\.json$")
_PIPELINE_CACHE_PATH = _CACHE_DIR / "pipeline_cache.json"
_PORTFOLIO_SCREEN_PATH = _CACHE_DIR / "portfolio_screen.json"


def _discover_cache() -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """
    Return (sorted tickers, sorted quarters, in-memory cache map).

    Reads in this priority order (later sources overwrite earlier on conflict):
      1. per_quarter/*.json (NB08 detailed payloads, if any)
      2. <CACHE_DIR>/*.json matching TICKER_YYYYQN.json
      3. pipeline_cache.json (consolidated NB08 output) — wins on conflict
    """
    cache_map: Dict[str, Dict[str, Any]] = {}
    tickers: set = set()
    quarters: set = set()

    def _ingest(file_path: Path) -> None:
        m = _CACHE_FILE_RE.match(file_path.name)
        if not m:
            return
        try:
            with open(file_path) as fh:
                payload = json.load(fh)
            key = f"{m.group('ticker')}_{m.group('quarter')}"
            cache_map[key] = payload
            tickers.add(m.group("ticker"))
            quarters.add(m.group("quarter"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", file_path.name, exc)

    # NOTE: per_quarter/*.json is intentionally NOT scanned. The demo is a
    # microscope on the 12 curated pairs (6 tickers x Q4 2020 / Q4 2023) that
    # NB08 materialised into pipeline_cache.json. The per_quarter/ folder
    # contains stale full-universe payloads that we don't want in the dropdown.

    # Cache root *.json
    for f in _CACHE_DIR.glob("*.json"):
        _ingest(f)

    # Consolidated pipeline_cache.json (NB08 output) — wins on conflict
    if _PIPELINE_CACHE_PATH.exists():
        try:
            with open(_PIPELINE_CACHE_PATH) as fh:
                blob = json.load(fh)
            entries = blob.get("cache", blob) if isinstance(blob, dict) else {}
            for key, payload in entries.items():
                if not isinstance(payload, dict):
                    continue
                cache_map[key] = payload
                if "_" in key:
                    tk, qt = key.rsplit("_", 1)
                    tickers.add(tk)
                    quarters.add(qt)
        except Exception as exc:
            logger.warning("Could not load pipeline_cache.json: %s", exc)

    global _CACHE_IS_EMPTY
    _CACHE_IS_EMPTY = not cache_map
    if _CACHE_IS_EMPTY:
        logger.warning(
            "No demo cache found under %s. Use the Pipeline tab to run the "
            "cache stage after upstream parquets are ready.",
            _CACHE_DIR,
        )

    return sorted(tickers), sorted(quarters), cache_map


_DISCOVERED_TICKERS, _DISCOVERED_QUARTERS, _IN_MEMORY_CACHE = _discover_cache()

# Only show entities/quarters from cache or raw corpus — never a synthetic grid.
_TICKERS: List[str] = list(_DISCOVERED_TICKERS)
_QUARTERS: List[str] = list(_DISCOVERED_QUARTERS)
if not _TICKERS or not _QUARTERS:
    try:
        from demo.data_manager import scan_raw_universe
        _scan = scan_raw_universe(_PROJECT_ROOT / "data")
        if _scan:
            if not _TICKERS:
                _TICKERS = list(_scan[0])
            if not _QUARTERS:
                _QUARTERS = list(_scan[1])
    except Exception:
        pass

_QUARTERS_DISPLAY = [f"Q{q[5]} {q[:4]}" for q in _QUARTERS] if _QUARTERS else []
_QUARTER_MAP = {disp: raw for disp, raw in zip(_QUARTERS_DISPLAY, _QUARTERS)}

logger.info(
    "Demo cache: %d tickers, %d quarters, %d cached results",
    len(_TICKERS), len(_QUARTERS), len(_IN_MEMORY_CACHE),
)


# ===========================================================================
# Data loading / pipeline invocation helpers
# ===========================================================================

def _load_cached_result(ticker: str, quarter: str) -> Optional[Dict[str, Any]]:
    """Load a pre-computed pipeline result. In-memory first, then disk."""
    key = f"{ticker}_{quarter}"
    if key in _IN_MEMORY_CACHE:
        return _IN_MEMORY_CACHE[key]

    # Demo-only: cache root only. per_quarter/ is intentionally ignored.
    candidate = _CACHE_DIR / f"{key}.json"
    if candidate.exists():
        try:
            with open(candidate) as f:
                payload = json.load(f)
            _IN_MEMORY_CACHE[key] = payload
            return payload
        except Exception as exc:
            logger.warning("Failed to load cache %s: %s", candidate, exc)
    return None


def _save_cached_result(ticker: str, quarter: str, result: Dict[str, Any]) -> None:
    """Persist a pipeline result to the cache directory."""
    cache_file = _CACHE_DIR / f"{ticker}_{quarter}.json"
    try:
        serialisable = {
            k: v for k, v in result.items()
            if not isinstance(v, pd.DataFrame)
        }
        with open(cache_file, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        _IN_MEMORY_CACHE[f"{ticker}_{quarter}"] = serialisable
    except Exception as exc:
        logger.warning("Failed to save cache: %s", exc)


def _load_structured_transcript(ticker: str, quarter: str) -> Optional[Dict[str, Any]]:
    """
    Build a structured transcript dict from the raw CIQ parquet, ctype-aware.

    Returns None if the parquet is missing or no rows match. Used only as a
    live fallback when the demo cache is missing the requested pair.
    """
    raw_path = _PROJECT_ROOT / "data" / "raw" / "ciq_transcripts.parquet"
    if not raw_path.exists():
        return None

    try:
        df = pd.read_parquet(raw_path)
    except Exception as exc:
        logger.warning("Could not read CIQ parquet: %s", exc)
        return None

    # Match on ticker if present, else fall back to companyname
    mask = pd.Series(False, index=df.index)
    if "ticker" in df.columns:
        mask |= df["ticker"].astype(str).str.upper() == ticker.upper()
    if "companyname" in df.columns:
        mask |= df["companyname"].astype(str).str.upper().str.contains(
            ticker.upper(), na=False
        )

    # FIX #3: handle fiscalyear + fiscalquarter (CIQ schema), with quarter fallback
    if "fiscalyear" in df.columns and "fiscalquarter" in df.columns:
        qmask = (
            df["fiscalyear"].astype(str).str.replace(r"\.0$", "", regex=True)
            + "Q"
            + df["fiscalquarter"].astype(str).str.replace(r"\.0$", "", regex=True)
        ) == quarter
        mask &= qmask
    elif "quarter" in df.columns:
        mask &= df["quarter"].astype(str) == quarter

    sub = df[mask]
    if sub.empty:
        return None

    # Build ctype-bucketed components
    if "component_type_id" in sub.columns:
        sub = sub[sub["component_type_id"].isin([2, 3, 4])]
    if "componentorder" in sub.columns:
        sub = sub.sort_values("componentorder")

    components = []
    for _, row in sub.iterrows():
        text = str(row.get("componenttext") or "").strip()
        if not text:
            continue
        components.append({
            "component_type_id": int(row.get("component_type_id", 4)),
            "componenttext": text,
        })

    if not components:
        return None

    return {"components": components}


def _run_or_load_pipeline(ticker: str, quarter: str) -> Tuple[Dict[str, Any], bool]:
    """Load cached result if available, otherwise run the LangGraph structured pipeline.

    Returns (result_dict, from_cache).
    """
    cached = _load_cached_result(ticker, quarter)
    if cached is not None:
        logger.info("Loaded cached result for %s %s", ticker, quarter)
        return cached, True

    transcript = _load_structured_transcript(ticker, quarter)
    if transcript is None:
        logger.warning("No CIQ data for %s %s", ticker, quarter)
        return _unavailable_result(ticker, quarter), False

    try:
        # Try the standard graph location first, then fallback
        try:
            from src.agents.graph import build_graph
        except ImportError:
            from src.graph import build_graph  # type: ignore
        app = build_graph()
        initial_state = {
            "transcript": transcript,
            "company_id": ticker,
            "ticker": ticker,
            "fiscal_quarter": quarter,
            "errors": [],
        }
        result = app.invoke(initial_state)
        _save_cached_result(ticker, quarter, result)
        return result, False
    except Exception as exc:
        logger.error("Pipeline run failed for %s %s: %s", ticker, quarter, exc)
        return _unavailable_result(ticker, quarter, error=str(exc)), False


def _unavailable_result(ticker: str, quarter: str, error: str = "") -> Dict[str, Any]:
    """Return an explicit 'data unavailable' result — no fabricated numbers."""
    notice = f"No data available for {ticker} {quarter}."
    if error:
        notice += f" Pipeline error: {error}"
    notice += " Please pick a different company-quarter from the dropdown."
    return {
        "company_id": ticker,
        "ticker": ticker,
        "fiscal_quarter": quarter,
        "extracted_targets": [],
        "spacy_baseline_targets": [],
        "historical_targets": [],
        "continuity_results": {"maintained": [], "rephrased": [], "dropped": [], "details": {}},
        "classification_results": {
            "n_dropped": 0, "n_total": 0, "risk_score": 0.0,
            "dropped_financial": [], "dropped_non_financial": [],
            "persistent_dropped": [], "ephemeral_dropped": [],
        },
        "report": {
            "summary": "No data available for this company-quarter.",
            "risk_flag": "N/A",
            "risk_score": 0.0,
            "dropped_targets": [],
            "recommendation": "Select a different company-quarter from the dropdown.",
            "generated_at": "",
        },
        "errors": [notice],
    }


# ===========================================================================
# Portfolio data
# ===========================================================================

def _load_portfolio_screen_blob() -> Dict[str, List[Dict[str, Any]]]:
    """Load the portfolio_screen.json materialised by scripts/build_demo_cache.py."""
    if not _PORTFOLIO_SCREEN_PATH.exists():
        return {}
    try:
        with open(_PORTFOLIO_SCREEN_PATH) as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            return {}
        return {k: v for k, v in blob.items() if not k.startswith("_")}
    except Exception as exc:
        logger.warning("Could not load portfolio_screen.json: %s", exc)
        return {}


_PORTFOLIO_SCREEN_BLOB = _load_portfolio_screen_blob()


def _derive_risk_flag(risk_score: float) -> str:
    """Derive risk flag from score using thresholds calibrated to demo data."""
    if risk_score >= 0.15:
        return "HIGH"
    if risk_score >= 0.10:
        return "MEDIUM"
    return "LOW"


def _get_portfolio_data(quarter: str) -> pd.DataFrame:
    """
    Return a DataFrame of MT-score companies for the given quarter, sorted desc.

    FIX #4: trust portfolio_screen.json when present (canonical NB08 output);
    only rebuild from in-memory cache as a fallback. Always re-derive risk_flag
    from score if it's 'unknown' or missing, so Tab 2 isn't all greyed out.
    """
    rows: List[Dict[str, Any]] = []

    # Prefer canonical NB08 output
    if quarter in _PORTFOLIO_SCREEN_BLOB:
        for r in _PORTFOLIO_SCREEN_BLOB[quarter]:
            ticker = str(r.get("ticker", ""))
            risk_score = float(r.get("risk_score", 0.0) or 0.0)
            n_dropped = int(r.get("n_dropped", 0) or 0)
            mt_score = float(r.get("mt_score", 0.0) or 0.0)
            flag = str(r.get("risk_flag", "") or "")
            if flag.lower() in ("", "unknown", "n/a", "none"):
                flag = _derive_risk_flag(risk_score)
            rows.append({
                "rank": 0,
                "ticker": ticker,
                "company_name": ticker,
                "mt_score_llm": round(risk_score, 4),
                "mt_score_spacy": round(mt_score, 4),
                "n_dropped": n_dropped,
                "risk_flag": flag,
            })

    # Fallback: rebuild from in-memory cache
    if not rows:
        for key, payload in _IN_MEMORY_CACHE.items():
            if not key.endswith(f"_{quarter}"):
                continue
            ticker = key.rsplit("_", 1)[0]
            cls = payload.get("classification_results", {}) or {}
            report = payload.get("report", {}) or {}
            risk_score = float(cls.get("risk_score", report.get("risk_score", 0.0)) or 0.0)
            n_dropped = int(cls.get("n_dropped", len(report.get("dropped_targets", []))) or 0)
            flag = str(report.get("risk_flag", "") or "")
            if flag.lower() in ("", "unknown", "n/a", "none"):
                flag = _derive_risk_flag(risk_score)
            rows.append({
                "rank": 0,
                "ticker": ticker,
                "company_name": ticker,
                "mt_score_llm": round(risk_score, 4),
                "mt_score_spacy": 0.0,
                "n_dropped": n_dropped,
                "risk_flag": flag,
            })

    if not rows:
        return pd.DataFrame(
            columns=["rank", "ticker", "company_name", "mt_score_llm",
                     "mt_score_spacy", "n_dropped", "risk_flag"]
        )

    df = pd.DataFrame(rows).sort_values("mt_score_llm", ascending=False).head(20).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ===========================================================================
# UI helpers
# ===========================================================================

# FIX #6: cover unknown / N/A explicitly so the gauge doesn't render grey-on-coloured.
_RISK_COLOUR = {
    "HIGH": "#d9534f",
    "MEDIUM": "#f0ad4e",
    "LOW": "#5cb85c",
    "UNKNOWN": "#888888",
    "N/A": "#888888",
}
_RISK_LABEL = {
    "HIGH": "HIGH",
    "MEDIUM": "MEDIUM",
    "LOW": "LOW",
    "UNKNOWN": "UNKNOWN",
    "N/A": "N/A",
}


def _normalise_flag(flag: Any) -> str:
    s = str(flag or "").strip().upper()
    if s in _RISK_LABEL:
        return s
    if s in ("UNKNOWN", "NONE", ""):
        return "UNKNOWN"
    return s or "UNKNOWN"


def _risk_gauge_html(risk_score: float, risk_flag: str) -> str:
    """Render a simple risk score gauge/meter as HTML."""
    pct = min(max(risk_score * 100, 0), 100)
    flag = _normalise_flag(risk_flag)
    colour = _RISK_COLOUR.get(flag, "#888")
    label = _RISK_LABEL.get(flag, flag)

    return f"""
<div style="font-family: sans-serif; padding: 12px 0;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: {colour};"></span>
    <span style="font-size: 1.1em; font-weight: 600; color: {colour};">
      Risk Flag: {label}
    </span>
    <span style="color: #555;">Risk Score: <strong>{risk_score:.3f}</strong></span>
  </div>
  <div style="background: #e0e0e0; border-radius: 8px; height: 18px; width: 100%; overflow: hidden;">
    <div style="
      background: {colour};
      width: {pct:.1f}%;
      height: 100%;
      border-radius: 8px;
      transition: width 0.5s ease;
    "></div>
  </div>
  <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #888; margin-top: 3px;">
    <span>0.0 — LOW</span><span>0.10 — MEDIUM</span><span>0.15 — HIGH</span><span>1.0</span>
  </div>
</div>
"""


def _targets_to_df(targets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of target dicts to a display DataFrame.

    Supports three schemas:
      - NB03 LLM (Gemini): metric_name, raw_text, numerical_value, unit,
        temporal_framing, trend_direction, is_financial
      - NB02 spaCy baseline: target_text, numeric_value, entity_label,
        sentence, is_financial
      - Legacy display: metric_name, context, target_type
    """
    if not targets:
        return pd.DataFrame(columns=["Metric Name", "Type", "Context"])

    def _fmt_value(v: Any) -> str:
        """Compact formatting for big numbers (89500000000 -> '89.5B')."""
        if v is None or v == "":
            return ""
        try:
            n = float(v)
        except (TypeError, ValueError):
            return str(v)
        absn = abs(n)
        if absn >= 1e9:
            return f"{n/1e9:.2f}B"
        if absn >= 1e6:
            return f"{n/1e6:.2f}M"
        if absn >= 1e3:
            return f"{n/1e3:.1f}K"
        return f"{n:g}"

    rows = []
    for t in targets:
        # Metric name: try every known key.
        metric = (
            t.get("metric_name")
            or t.get("canonical_name")
            or t.get("target_text")     # spaCy
            or t.get("metric")
            or t.get("text")
            or ""
        )

        # Type: explicit field, else derive from is_financial.
        ttype = (
            t.get("target_type")
            or t.get("type")
            or ("financial" if t.get("is_financial") else "non-financial")
        )

        # Context: build from richest available fields per schema.
        context_parts: List[str] = []

        # LLM-style numeric: numerical_value + unit
        nv = t.get("numerical_value")
        unit = t.get("unit") or ""
        if nv is not None:
            val = _fmt_value(nv)
            if val:
                if unit and "USD_millions" in unit:
                    context_parts.append(f"${val}")
                elif unit and "USD" in unit:
                    context_parts.append(f"${val}")
                elif unit and unit not in ("", "none", "None"):
                    context_parts.append(f"{val} {unit}")
                else:
                    context_parts.append(val)

        # spaCy-style: numeric_value is already a string like "$89.5 billion"
        nv_str = t.get("numeric_value")
        if not nv and nv_str:
            context_parts.append(str(nv_str))

        # Direction (LLM)
        if t.get("trend_direction"):
            context_parts.append(str(t["trend_direction"]))
        elif t.get("direction"):
            context_parts.append(str(t["direction"]))

        # Temporal framing (LLM) or sentence snippet (spaCy)
        if t.get("temporal_framing"):
            context_parts.append(str(t["temporal_framing"]).replace("_", " "))
        if t.get("period"):
            context_parts.append(str(t["period"]))

        # Fallback: raw_text (LLM) or sentence (spaCy) or context (legacy)
        if not context_parts:
            fallback = (
                t.get("raw_text")
                or t.get("sentence")
                or t.get("context")
                or ""
            )
            if fallback:
                context_parts.append(str(fallback))

        context = " · ".join(p for p in context_parts if p)

        rows.append({
            "Metric Name": str(metric),
            "Type": str(ttype),
            "Context": str(context)[:160],
        })
    return pd.DataFrame(rows)


def _dropped_to_df(dropped_targets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert dropped target records to a display DataFrame."""
    if not dropped_targets:
        return pd.DataFrame(columns=["Target Name", "Last Seen", "Type", "Persistence"])
    rows = []
    for t in dropped_targets:
        last_seen = (
            t.get("last_seen_quarter")
            or t.get("last_quarter")
            or t.get("last_seen")
            or ""
        )
        # Empty string is common when continuity tracking didn't record
        # the prior quarter; show a clearer placeholder for the demo.
        if not str(last_seen).strip():
            last_seen = "prior quarter"

        ttype = (
            t.get("type")
            or t.get("target_type")
            or ("financial" if t.get("is_financial") else "non-financial")
        )

        persistence = (
            t.get("persistence")
            or t.get("persistent_count")
            or t.get("n_consecutive_quarters")
            or ""
        )

        rows.append({
            "Target Name": (
                t.get("target_name")
                or t.get("metric_name")
                or t.get("canonical_name")
                or t.get("metric")
                or t.get("name")
                or ""
            ),
            "Last Seen": str(last_seen),
            "Type": str(ttype),
            "Persistence": str(persistence),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Tab 1 — Company Analysis
# ===========================================================================

def analyse_company(
    ticker: str,
    quarter_display: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Any, ...]:
    """Gradio event handler for the Company Analysis tab."""
    quarter = _QUARTER_MAP.get(quarter_display, quarter_display)

    # FIX #1: regexes use single backslashes inside raw strings.
    if not re.match(r"^\d{4}Q[1-4]$", str(quarter)):
        m = re.match(r"^Q([1-4])\s+(\d{4})$", str(quarter_display).strip())
        if m:
            quarter = f"{m.group(2)}Q{m.group(1)}"

    progress(0.1, desc="Loading pipeline result…")

    result, from_cache = _run_or_load_pipeline(ticker, quarter)

    progress(0.6, desc="Processing results…")

    extracted = result.get("extracted_targets", []) or []
    spacy_targets = result.get("spacy_baseline_targets", []) or []
    report = result.get("report", {}) or {}
    cls = result.get("classification_results", {}) or {}
    errors = result.get("errors", []) or []

    current_df = _targets_to_df(extracted)

    dropped = report.get("dropped_targets") or (
        (cls.get("dropped_financial") or []) + (cls.get("dropped_non_financial") or [])
    )
    dropped_df = _dropped_to_df(dropped)

    risk_score = float(cls.get("risk_score", report.get("risk_score", 0.0)) or 0.0)
    risk_flag = _normalise_flag(report.get("risk_flag"))
    if risk_flag == "UNKNOWN" and risk_score > 0:
        # Derive a flag from score so the gauge isn't grey on a real result.
        risk_flag = _derive_risk_flag(risk_score)

    gauge_html = _risk_gauge_html(risk_score, risk_flag)

    spacy_df = _targets_to_df(spacy_targets)
    llm_df = _targets_to_df(extracted)

    summary = report.get("summary", "No report generated.")
    recommendation = report.get("recommendation", "")
    cache_note = " *(from cache)*" if from_cache else " *(live run)*"
    narrative_md = f"""
## Risk Assessment — {ticker} {quarter}{cache_note}

**Risk Flag:** {_RISK_LABEL.get(risk_flag, risk_flag)}

{summary}

---

**Recommendation:** {recommendation}
"""

    # FIX #2: real newlines.
    if errors:
        errors_md = "**Notices:**\n" + "\n".join(f"- {e}" for e in errors[:5])
    else:
        errors_md = "*No errors.*"

    progress(1.0, desc="Done")
    return current_df, dropped_df, gauge_html, spacy_df, llm_df, narrative_md, errors_md


# ===========================================================================
# Tab 2 — Portfolio Screen
# ===========================================================================

def load_portfolio(quarter_display: str) -> Tuple[pd.DataFrame, str]:
    """Load top-20 MT-score companies for a selected quarter."""
    quarter = _QUARTER_MAP.get(quarter_display, quarter_display)
    df = _get_portfolio_data(quarter)
    if df.empty:
        status = (
            f"No watchlist data for **{quarter_display}**. "
            "Run the **cache** stage on Pipeline after upstream steps complete."
        )
    else:
        status = f"Showing top {len(df)} companies by LLM MT score for **{quarter_display}**."
    return df, status


# ===========================================================================
# Workflow — in-app pipeline control (no CLI)
# ===========================================================================

from demo.pipeline_runner import WORKFLOW_STAGES, run_stage_streaming


def _refresh_status_markdown() -> str:
    """Render the current pipeline-state table for the Workflow tab."""
    try:
        from src.status import describe_pipeline_status
        text = describe_pipeline_status(_PROJECT_ROOT / "data")
    except Exception as exc:
        text = f"Could not read status: {exc}"
    return f"```\n{text}\n```"


def _cache_banner_markdown() -> str:
    if _CACHE_IS_EMPTY:
        return (
            "> **No analysis cache yet.** Upload corpus files on **Data**, then run "
            "pipeline stages on **Pipeline** (or click **Run all**). Results appear "
            "on **Entity Report** and **Watchlist** automatically."
        )
    return ""


def _workflow_intro_markdown() -> str:
    from demo.bootstrap import ensure_ready

    lines = ensure_ready(_PROJECT_ROOT)
    setup = "\n".join(f"- {line}" for line in lines) if lines else "- Environment ready."
    return f"""
### Workflow

1. **Download data manually** and place files under `data/raw/` (at minimum
   `ciq_transcripts.parquet` or `transcripts.parquet`, plus any other parquets
   you use for evaluation).
2. **Run stages below** — everything happens inside this app; no terminal commands.
3. Open **Company Analysis** or **Portfolio Screen** once the cache stage finishes.

{setup}
"""


def _reload_cache_choices():
    """Re-read cache from disk and return dropdown updates for the UI."""
    global _DISCOVERED_TICKERS, _DISCOVERED_QUARTERS, _IN_MEMORY_CACHE, _CACHE_IS_EMPTY
    global _TICKERS, _QUARTERS, _QUARTERS_DISPLAY, _QUARTER_MAP

    tickers, quarters, cache = _discover_cache()
    _DISCOVERED_TICKERS = tickers
    _DISCOVERED_QUARTERS = quarters
    _IN_MEMORY_CACHE = cache

    _TICKERS = tickers or _TICKERS
    _QUARTERS = quarters or _QUARTERS
    if not _TICKERS or not _QUARTERS:
        try:
            from demo.data_manager import scan_raw_universe
            _scan = scan_raw_universe(_PROJECT_ROOT / "data")
            if _scan:
                if not _TICKERS:
                    _TICKERS = list(_scan[0])
                if not _QUARTERS:
                    _QUARTERS = list(_scan[1])
        except Exception:
            pass
    _QUARTERS_DISPLAY = [f"Q{q[5]} {q[:4]}" for q in _QUARTERS]
    _QUARTER_MAP = {disp: raw for disp, raw in zip(_QUARTERS_DISPLAY, _QUARTERS)}

    default_ticker = _TICKERS[0] if _TICKERS else None
    default_quarter_disp = _QUARTERS_DISPLAY[-1] if _QUARTERS_DISPLAY else None

    banner_update = (
        gr.update(value="", visible=False)
        if not _CACHE_IS_EMPTY
        else gr.update(value=_cache_banner_markdown(), visible=True)
    )

    return (
        gr.update(choices=_TICKERS, value=default_ticker),
        gr.update(choices=_QUARTERS_DISPLAY, value=default_quarter_disp),
        gr.update(choices=_QUARTERS_DISPLAY, value=default_quarter_disp),
        _refresh_status_markdown(),
        banner_update,
    )


def _run_workflow_stage(stage: str, extra_args: str, api_key: str):
    """Generator wrapper used by the Workflow tab Run button."""
    yield from run_stage_streaming(stage, extra_args, api_key=api_key or "")


def _run_all_stages(extra_args: str, api_key: str):
    yield from _run_workflow_stage("all", extra_args, api_key)


def drill_down_report(portfolio_df: pd.DataFrame, evt: gr.SelectData) -> str:
    """Generate a drill-down summary for the selected portfolio row."""
    try:
        row_idx = evt.index[0]
        row = portfolio_df.iloc[row_idx]
        ticker = row.get("ticker", "")
        mt_llm = float(row.get("mt_score_llm", 0) or 0)
        mt_spacy = float(row.get("mt_score_spacy", 0) or 0)
        n_dropped = int(row.get("n_dropped", 0) or 0)
        risk_flag = _normalise_flag(row.get("risk_flag", "LOW"))
        company = row.get("company_name", ticker)

        colour = _RISK_COLOUR.get(risk_flag, "#888")
        return f"""
## {company} ({ticker})

| Field | Value |
|-------|-------|
| **LLM MT Score** | {mt_llm:.3f} |
| **spaCy MT Score** | {mt_spacy:.3f} |
| **Targets Dropped** | {n_dropped} |
| **Risk Flag** | <span style="color:{colour}">**{risk_flag}**</span> |

*Open the **Entity Report** tab and select this entity for the full analysis.*
"""
    except Exception as exc:
        return f"*Could not load drill-down: {exc}*"


# ===========================================================================
# Overview dashboard
# ===========================================================================

def _overview_stats() -> Dict[str, Any]:
    """Collect KPI numbers for the Overview tab."""
    from demo.data_manager import _transcripts_ready

    data_dir = _PROJECT_ROOT / "data"
    pipeline_stage = "Not started"
    pipeline_detail = "awaiting corpus"
    try:
        from src.status import describe_pipeline_status
        status_text = describe_pipeline_status(data_dir)
        first_line = status_text.strip().split("\n")[0] if status_text else ""
        if first_line:
            pipeline_stage = first_line[:40]
            pipeline_detail = "see Pipeline tab"
    except Exception:
        pass

    if _transcripts_ready(data_dir) and _CACHE_IS_EMPTY:
        pipeline_detail = "corpus ready — run pipeline"

    status_message = ""
    if _CACHE_IS_EMPTY:
        if not _transcripts_ready(data_dir):
            status_message = (
                "Upload a transcript parquet on the Data tab to begin. "
                "No synthetic demo data is shown."
            )
        else:
            status_message = (
                "Corpus detected. Run pipeline stages (baseline → llm → rag → "
                "calibrate → cache) to populate analysis results."
            )

    return {
        "n_tickers": len(_TICKERS),
        "n_quarters": len(_QUARTERS),
        "cache_pairs": len(_IN_MEMORY_CACHE),
        "data_ready": _transcripts_ready(data_dir),
        "pipeline_stage": pipeline_stage,
        "pipeline_detail": pipeline_detail,
        "status_message": status_message,
    }


def render_overview_html() -> str:
    """Render the Overview tab HTML dashboard."""
    from demo.data_manager import render_overview_dashboard

    stats = _overview_stats()
    quarter = _QUARTERS[-1] if _QUARTERS else ""
    quarter_disp = _QUARTERS_DISPLAY[-1] if _QUARTERS_DISPLAY else "—"
    rows: List[Dict[str, Any]] = []
    if quarter:
        df = _get_portfolio_data(quarter)
        if not df.empty:
            rows = df.to_dict(orient="records")

    return render_overview_dashboard(
        _PROJECT_ROOT / "data",
        stats,
        rows,
        quarter_disp,
    )


# ===========================================================================
# Gradio interface assembly
# ===========================================================================

def build_interface() -> "gr.Blocks":
    """Build and return the full Gradio Blocks interface."""
    from demo.interface import build_interface as _build
    return _build()


# ===========================================================================
# Entry point
# ===========================================================================

def main(argv=None) -> int:
    """Launch the local Gradio app from a normal Python process."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch Target Tracker Gradio app")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args(argv)

    from demo.interface import build_interface
    from demo.theme import APP_CSS, build_theme

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=build_theme(),
        css=APP_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
