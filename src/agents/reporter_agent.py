"""
reporter_agent.py — Reporter Agent node for the EarningsLens LangGraph pipeline.

This is the final agent node. It consumes the classification results and
generates a structured narrative report using an LLM, covering:
    a) Summary: one-paragraph risk assessment
    b) Dropped targets table: target name, last seen quarter, type, persistence
    c) Risk flag: HIGH / MEDIUM / LOW based on risk_score thresholds
    d) Recommendation: brief investment implication statement

The output is written to ``state.report`` as a structured JSON-serialisable dict.

Risk flag thresholds
--------------------
    risk_score > 0.6  → HIGH
    0.3 ≤ risk_score ≤ 0.6 → MEDIUM
    risk_score < 0.3  → LOW

LLM backend
-----------
The reporter uses the same env-var configuration scheme as the extractor
agent so a single Google API key drives the whole pipeline. The key
resolution order is:

    EARNINGSLENS_LLM_API_KEY  →  GOOGLE_API_KEY  →  GEMINI_API_KEY  →  OPENAI_API_KEY

Backend / model / base_url come from EARNINGSLENS_LLM_BACKEND,
EARNINGSLENS_LLM_MODEL, and EARNINGSLENS_LLM_BASE_URL. Defaults target
``gemini-2.5-flash-lite`` against Gemini's OpenAI-compatible endpoint.

When no key is found OR the LLM call fails, the reporter SILENTLY falls
back to the deterministic template-based summary / recommendation. This
keeps the report node side-effect-free with respect to ``state.errors``,
which is how Reporter behaviour was intended pre-Google migration.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .state import PipelineState

logger = logging.getLogger(__name__)

# Risk thresholds
_HIGH_THRESHOLD = 0.6
_LOW_THRESHOLD = 0.3

# System prompt for the LLM reporter
_SYSTEM_PROMPT = """You are a financial analyst specialising in earnings call analysis and 
forward-guidance tracking. Your task is to produce a concise, structured report 
assessing the risk implied by a company dropping previously stated management targets.

Always respond with a valid JSON object following exactly the schema provided.
Do not include any text outside the JSON block."""

_REPORT_PROMPT_TEMPLATE = """
Company: {ticker} ({company_id})
Fiscal Quarter: {quarter}
Risk Score: {risk_score:.3f} ({risk_flag})

Dropped Targets Summary:
- Total targets analysed: {n_total}
- Dropped: {n_dropped} ({pct_dropped:.0%})
- Of dropped — financial: {n_dropped_financial}, non-financial: {n_dropped_nonfinancial}
- Persistent drops (appeared every quarter for 12+ quarters): {n_persistent}

Dropped Target Details:
{dropped_table}

Please produce a structured report with the following JSON schema:
{{
  "summary": "<one-paragraph risk assessment discussing what targets were dropped, their significance, and what this implies about management's forward guidance transparency>",
  "recommendation": "<2-3 sentence investment implication statement — should investors be concerned? what should they watch for next quarter?>",
  "key_concerns": ["<concern 1>", "<concern 2>", "<concern 3 or more as needed>"]
}}
"""


# ---------------------------------------------------------------------------
# LLM config helpers (mirrors extractor_agent._resolve_llm_kwargs)
# ---------------------------------------------------------------------------

def _env(name: str) -> Optional[str]:
    """Return env var value, treating empty string as not-set."""
    v = os.environ.get(name)
    return v if v else None


def _resolve_llm_config() -> Optional[Dict[str, Any]]:
    """
    Build a config dict for an OpenAI-compatible chat completion.

    Returns None when no API key is found anywhere — in which case the
    reporter silently falls back to its deterministic template path.
    """
    api_key = (
        _env("EARNINGSLENS_LLM_API_KEY")
        or _env("GOOGLE_API_KEY")
        or _env("GEMINI_API_KEY")
        or _env("OPENAI_API_KEY")
    )
    if not api_key:
        return None

    backend = _env("EARNINGSLENS_LLM_BACKEND") or "openai"
    model = _env("EARNINGSLENS_LLM_MODEL") or "gemini-2.5-flash-lite"
    base_url = (
        _env("EARNINGSLENS_LLM_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    return {
        "backend": backend,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


# ---------------------------------------------------------------------------
# Reporter node
# ---------------------------------------------------------------------------

def reporter_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: generate structured narrative report from classification results.

    Execution flow
    --------------
    1. Extract classification_results from state.
    2. Determine risk_flag from risk_score.
    3. Attempt LLM generation of narrative components (silent fallback if
       no API key or call fails).
    4. Otherwise, use a deterministic template-based fallback.
    5. Write structured report dict to state.

    Parameters
    ----------
    state : PipelineState
        Must contain ``classification_results``, ``company_id``,
        ``ticker``, and ``fiscal_quarter``.

    Returns
    -------
    PipelineState: Updated state with ``report``.
    """
    company_id = state.get("company_id", "UNKNOWN")
    ticker = state.get("ticker", company_id)
    quarter = state.get("fiscal_quarter", "")
    cls_results: Dict[str, Any] = state.get("classification_results", {}) or {}
    errors: List[str] = list(state.get("errors", []))

    risk_score = float(cls_results.get("risk_score", 0.0))
    risk_flag = _score_to_flag(risk_score)
    n_total = int(cls_results.get("n_total", 0))
    n_dropped = int(cls_results.get("n_dropped", 0))
    dropped_financial = cls_results.get("dropped_financial", []) or []
    dropped_non_financial = cls_results.get("dropped_non_financial", []) or []
    persistent_dropped = cls_results.get("persistent_dropped", []) or []

    logger.info(
        "[%s] reporter_node: risk_score=%.3f risk_flag=%s n_dropped=%d",
        company_id,
        risk_score,
        risk_flag,
        n_dropped,
    )

    # -----------------------------------------------------------------------
    # Build dropped targets table (list of dicts for report)
    # -----------------------------------------------------------------------
    all_dropped = dropped_financial + [
        t for t in dropped_non_financial
        if t not in dropped_financial
    ]
    persistent_names = {t.get("metric_name", "") for t in persistent_dropped}

    dropped_table_rows: List[Dict[str, Any]] = []
    for target in all_dropped:
        row = {
            "target_name": target.get("metric_name", ""),
            "last_seen_quarter": target.get("last_seen_quarter", ""),
            "type": target.get("target_type", "financial"),
            "persistence": "persistent" if target.get("metric_name", "") in persistent_names else "ephemeral",
        }
        dropped_table_rows.append(row)

    # -----------------------------------------------------------------------
    # Attempt LLM narrative generation (silent fallback on any failure)
    # -----------------------------------------------------------------------
    llm_summary = ""
    llm_recommendation = ""
    llm_key_concerns: List[str] = []

    if n_dropped > 0:
        cfg = _resolve_llm_config()
        if cfg is not None:
            try:
                llm_summary, llm_recommendation, llm_key_concerns = _generate_llm_narrative(
                    cfg=cfg,
                    company_id=company_id,
                    ticker=ticker,
                    quarter=quarter,
                    risk_score=risk_score,
                    risk_flag=risk_flag,
                    n_total=n_total,
                    n_dropped=n_dropped,
                    dropped_financial=dropped_financial,
                    dropped_non_financial=dropped_non_financial,
                    persistent_dropped=persistent_dropped,
                    dropped_table_rows=dropped_table_rows,
                )
            except Exception as exc:
                # Log only — don't pollute state.errors. The deterministic
                # template fallback below produces a perfectly serviceable
                # report and the LLM narrative is purely additive value.
                logger.warning(
                    "[%s] reporter_node: LLM narrative generation failed (using template): %s",
                    company_id,
                    exc,
                )
        else:
            logger.info(
                "[%s] reporter_node: no LLM API key configured; using template narrative",
                company_id,
            )

    # Fallback if LLM failed, was disabled, or nothing was dropped
    if not llm_summary:
        llm_summary = _template_summary(
            company_id=company_id,
            ticker=ticker,
            quarter=quarter,
            risk_score=risk_score,
            risk_flag=risk_flag,
            n_total=n_total,
            n_dropped=n_dropped,
            n_persistent=len(persistent_dropped),
            n_financial=len(dropped_financial),
            n_nonfinancial=len(dropped_non_financial),
        )

    if not llm_recommendation:
        llm_recommendation = _template_recommendation(risk_flag, n_dropped, persistent_dropped)

    # -----------------------------------------------------------------------
    # Assemble final report
    # -----------------------------------------------------------------------
    report: Dict[str, Any] = {
        "summary": llm_summary,
        "risk_flag": risk_flag,
        "risk_score": risk_score,
        "dropped_targets": dropped_table_rows,
        "recommendation": llm_recommendation,
        "key_concerns": llm_key_concerns,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "company_id": company_id,
            "ticker": ticker,
            "quarter": quarter,
            "n_total": n_total,
            "n_dropped": n_dropped,
            "n_dropped_financial": len(dropped_financial),
            "n_dropped_non_financial": len(dropped_non_financial),
            "n_persistent_dropped": len(persistent_dropped),
        },
    }

    logger.info(
        "[%s] Report generated: risk_flag=%s, %d dropped targets",
        company_id,
        risk_flag,
        len(dropped_table_rows),
    )

    return {**state, "report": report, "errors": errors}


# ---------------------------------------------------------------------------
# Risk flag mapping
# ---------------------------------------------------------------------------

def _score_to_flag(risk_score: float) -> str:
    """
    Map a numeric risk score to a categorical flag.

    Parameters
    ----------
    risk_score: Float in [0, 1].

    Returns
    -------
    str: "HIGH", "MEDIUM", or "LOW".
    """
    if risk_score > _HIGH_THRESHOLD:
        return "HIGH"
    elif risk_score >= _LOW_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# LLM narrative generation
# ---------------------------------------------------------------------------

def _generate_llm_narrative(
    cfg: Dict[str, Any],
    company_id: str,
    ticker: str,
    quarter: str,
    risk_score: float,
    risk_flag: str,
    n_total: int,
    n_dropped: int,
    dropped_financial: List[Dict[str, Any]],
    dropped_non_financial: List[Dict[str, Any]],
    persistent_dropped: List[Dict[str, Any]],
    dropped_table_rows: List[Dict[str, Any]],
) -> Tuple[str, str, List[str]]:
    """
    Use the LLM to generate the summary and recommendation narratives.

    Uses a direct OpenAI-compatible chat-completions call (NOT
    LLMTargetExtractor — that class is for target extraction and does not
    expose a generic chat method). Targets the same Gemini endpoint the
    extractor uses by default.

    Returns
    -------
    Tuple[str, str, List[str]]: (summary, recommendation, key_concerns).
    """
    # Local import so the reporter module doesn't hard-require the openai
    # package at import time. If the package is missing, this raises and
    # the caller silently falls back to the template path.
    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": cfg["api_key"]}
    base_url = cfg.get("base_url")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    pct_dropped = n_dropped / n_total if n_total > 0 else 0.0

    # Format dropped table for prompt
    table_lines = []
    for row in dropped_table_rows[:20]:  # limit prompt length
        table_lines.append(
            f"  - {row['target_name']} | last seen: {row['last_seen_quarter']} "
            f"| type: {row['type']} | {row['persistence']}"
        )
    dropped_table_str = "\n".join(table_lines) if table_lines else "  (none)"

    prompt = _REPORT_PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_id=company_id,
        quarter=quarter,
        risk_score=risk_score,
        risk_flag=risk_flag,
        n_total=n_total,
        n_dropped=n_dropped,
        pct_dropped=pct_dropped,
        n_dropped_financial=len(dropped_financial),
        n_dropped_nonfinancial=len(dropped_non_financial),
        n_persistent=len(persistent_dropped),
        dropped_table=dropped_table_str,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Gemini's OpenAI-compatible endpoint accepts response_format on most
    # current models, but not all of them; build kwargs defensively.
    completion_kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.0,
    }

    try:
        # Try with JSON-object response format first (cleaner parsing).
        response = client.chat.completions.create(
            **completion_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Fall back to plain completion if response_format is unsupported.
        response = client.chat.completions.create(**completion_kwargs)

    raw = (response.choices[0].message.content or "").strip()

    # Gemini sometimes wraps JSON in ```json ... ``` fences; strip them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # After stripping backticks, the first line may be "json"
        first_nl = raw.find("\n")
        if first_nl != -1 and raw[:first_nl].strip().lower() in {"json", ""}:
            raw = raw[first_nl + 1:]
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    parsed = json.loads(raw) if raw else {}
    summary = str(parsed.get("summary", "") or "").strip()
    recommendation = str(parsed.get("recommendation", "") or "").strip()
    key_concerns_raw = parsed.get("key_concerns", []) or []
    key_concerns = [str(k).strip() for k in key_concerns_raw if str(k).strip()]
    return summary, recommendation, key_concerns


# ---------------------------------------------------------------------------
# Fallback template generation
# ---------------------------------------------------------------------------

def _template_summary(
    company_id: str,
    ticker: str,
    quarter: str,
    risk_score: float,
    risk_flag: str,
    n_total: int,
    n_dropped: int,
    n_persistent: int,
    n_financial: int,
    n_nonfinancial: int,
) -> str:
    """Generate a deterministic summary narrative when LLM is unavailable."""
    pct = f"{n_dropped / n_total * 100:.0f}%" if n_total > 0 else "N/A"
    if n_dropped == 0:
        return (
            f"{ticker} ({company_id}) showed high guidance continuity in {quarter}, "
            f"with all {n_total} previously stated targets maintained or restated. "
            f"The Moving Targets risk score is {risk_score:.3f} ({risk_flag}), "
            f"suggesting low obfuscation risk."
        )
    return (
        f"{ticker} ({company_id}) dropped {n_dropped} of {n_total} previously stated "
        f"management targets in {quarter} ({pct} attrition rate). "
        f"Of these, {n_financial} were financial targets and {n_nonfinancial} were "
        f"non-financial targets. {n_persistent} dropped target(s) were persistent, "
        f"having appeared in every quarter for at least three years. "
        f"The Moving Targets risk score is {risk_score:.3f} ({risk_flag}), "
        f"indicating {'elevated' if risk_flag != 'LOW' else 'moderate'} obfuscation risk."
    )


def _template_recommendation(
    risk_flag: str,
    n_dropped: int,
    persistent_dropped: List[Dict[str, Any]],
) -> str:
    """Generate a deterministic recommendation when LLM is unavailable."""
    if risk_flag == "HIGH":
        return (
            "The high Moving Targets score warrants caution. Investors should probe "
            "management on the abandoned guidance metrics and monitor whether "
            "performance on those dimensions has deteriorated. Consider reviewing "
            "short-term position sizing."
        )
    elif risk_flag == "MEDIUM":
        return (
            "The moderate Moving Targets score suggests selective target attrition. "
            "Investors should seek clarification on the dropped metrics during the "
            "Q&A session and track them in subsequent quarters."
        )
    else:
        return (
            "The low Moving Targets score indicates strong guidance continuity. "
            "No immediate action is required, but ongoing monitoring is recommended."
        )
