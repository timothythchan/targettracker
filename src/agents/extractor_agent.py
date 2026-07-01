"""
extractor_agent.py — Extractor Agent node for the Target Tracker LangGraph pipeline.

This agent is the first node in the pipeline. It extracts forward-looking
guidance targets from the earnings call transcript using the LLM-based extractor.

Extraction call paths
---------------------
The LLM extractor exposes two ways to process a transcript, and they are
NOT equivalent:

  * ``extract_transcript(transcript_dict)`` — buckets components by their
    real CIQ ``component_type`` (2 = Presentation, 3 = Analyst Question,
    4 = Management Answer), runs one LLM call per bucket with the matching
    prompt, stamps every output target with the correct ``component_type``,
    and runs a final transcript-level dedup.

  * ``extract_targets(text, component_type=0)`` — flat-string path used only
    when structured components are unavailable.

This node prefers the structured path whenever a ``transcript`` dict is in state.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, List, Optional

from .state import PipelineState

logger = logging.getLogger(__name__)


def _import_llm_extractor():
    """Lazily import LLMTargetExtractor to avoid circular imports."""
    try:
        from src.llm_extraction.llm_extractor import LLMTargetExtractor
        return LLMTargetExtractor
    except ImportError as exc:
        logger.warning("Could not import LLMTargetExtractor: %s", exc)
        return None


def _resolve_llm_kwargs() -> Dict[str, Any]:
    """Build kwargs for ``LLMTargetExtractor(...)`` from environment variables."""
    def _env(name: str) -> Optional[str]:
        v = os.environ.get(name)
        return v if v else None

    backend = _env("EARNINGSLENS_LLM_BACKEND") or "openai"
    model = _env("EARNINGSLENS_LLM_MODEL") or "gemini-2.5-flash-lite"
    base_url = (
        _env("EARNINGSLENS_LLM_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    api_key = (
        _env("EARNINGSLENS_LLM_API_KEY")
        or _env("GOOGLE_API_KEY")
        or _env("GEMINI_API_KEY")
        or _env("OPENAI_API_KEY")
    )

    return {
        "backend": backend,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _structured_transcript_from_state(state: PipelineState) -> Optional[Dict[str, Any]]:
    """Normalise ``state['transcript']`` into the dict ``extract_transcript`` expects."""
    raw = state.get("transcript")
    if not isinstance(raw, dict):
        return None

    components_in = raw.get("components") or []
    if not components_in:
        return None

    normalised: List[Dict[str, Any]] = []
    for comp in components_in:
        if not isinstance(comp, dict):
            continue
        text = (
            comp.get("text")
            or comp.get("componenttext")
            or comp.get("component_text")
            or ""
        ).strip()
        if not text:
            continue
        ctype = int(
            comp.get("component_type")
            or comp.get("component_type_id")
            or comp.get("ctype")
            or 0
        )
        normalised.append({"text": text, "component_type": ctype})

    if not normalised:
        return None

    return {"components": normalised}


async def extractor_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: extract forward-looking targets from the transcript via LLM.

    Uses ``pre_extracted_targets`` from cache builds when present to skip live
    API calls. Returns empty ``extracted_targets`` when no API key is available
    and no pre-extracted targets exist.
    """
    transcript_text = state.get("transcript_text", "")
    company_id = state.get("company_id", "UNKNOWN")
    quarter = state.get("fiscal_quarter", "")
    errors: List[str] = list(state.get("errors", []))

    structured = _structured_transcript_from_state(state)

    if not transcript_text and not structured:
        msg = f"[{company_id}] extractor_node: no transcript_text and no structured transcript"
        logger.error(msg)
        errors.append(msg)
        return {**state, "extracted_targets": [], "errors": errors}

    if not transcript_text and structured:
        transcript_text = "\n\n".join(
            (c.get("text") or "").strip()
            for c in structured.get("components", [])
            if (c.get("text") or "").strip()
        )

    logger.info(
        "[%s] extractor_node: processing %d chars for %s (structured=%s)",
        company_id,
        len(transcript_text),
        quarter,
        bool(structured),
    )

    llm_targets: List[Dict[str, Any]] = []
    llm_succeeded = False

    pre_extracted = state.get("pre_extracted_targets")
    if pre_extracted:
        try:
            pre_extracted_list = list(pre_extracted)
        except TypeError:
            pre_extracted_list = []
        if pre_extracted_list:
            logger.info(
                "[%s] extractor_node: using %d pre_extracted_targets, skipping LLM call",
                company_id,
                len(pre_extracted_list),
            )
            llm_targets = pre_extracted_list
            llm_succeeded = True

    LLMExtractorClass = _import_llm_extractor() if not llm_succeeded else None

    if LLMExtractorClass is not None:
        llm_kwargs = _resolve_llm_kwargs()
        if not llm_kwargs.get("api_key"):
            msg = (
                f"[{company_id}] LLMTargetExtractor: no API key found "
                "(set GOOGLE_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)"
            )
            logger.warning(msg)
            errors.append(msg)
        else:
            try:
                llm_extractor = LLMExtractorClass(**llm_kwargs)

                if structured is not None:
                    extract_call = llm_extractor.extract_transcript(structured)
                else:
                    warn_msg = (
                        f"[{company_id}] LLM extractor falling back to flat "
                        "extract_targets (component_type=0) — structured transcript missing."
                    )
                    logger.warning(warn_msg)
                    errors.append(warn_msg)
                    extract_call = llm_extractor.extract_targets(
                        transcript_text, component_type=0
                    )

                if inspect.iscoroutine(extract_call):
                    llm_targets = await extract_call
                else:
                    llm_targets = extract_call

                if llm_targets:
                    llm_succeeded = True
                    logger.info(
                        "[%s] LLM extractor (%s): found %d targets",
                        company_id,
                        llm_kwargs.get("model"),
                        len(llm_targets),
                    )
                else:
                    empty_msg = f"[{company_id}] LLM extractor returned empty results"
                    logger.warning(empty_msg)
                    errors.append(empty_msg)
            except Exception as exc:
                msg = f"[{company_id}] LLMTargetExtractor failed: {exc}"
                logger.warning(msg)
                errors.append(msg)
    elif not llm_succeeded:
        msg = f"[{company_id}] LLMTargetExtractor not available"
        logger.warning(msg)
        errors.append(msg)

    return {
        **state,
        "extracted_targets": llm_targets if llm_succeeded else [],
        "errors": errors,
    }
