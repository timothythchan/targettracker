"""
extractor_agent.py — Extractor Agent node for the EarningsLens LangGraph pipeline.

This agent is the first node in the pipeline. It extracts forward-looking
guidance targets from the earnings call transcript using the LLM-based extractor
as the primary method, with automatic fallback to the spaCy baseline extractor
if the LLM call fails or returns empty results.

Both extractors are always run when the LLM extractor succeeds so that spaCy
baseline targets are available in state for downstream comparison.

Extraction call paths
---------------------
The LLM extractor exposes two ways to process a transcript, and they are
NOT equivalent:

  * ``extract_transcript(transcript_dict)`` — buckets components by their
    real CIQ ``component_type`` (2 = Presentation, 3 = Analyst Question,
    4 = Management Answer), runs one LLM call per bucket with the matching
    prompt, stamps every output target with the correct ``component_type``,
    and runs a final transcript-level dedup. THIS is the path NB03 v2 used
    to produce ``llm_targets.parquet`` (the source of NB04's 0.1455 strict
    MT reference for AT&T 2023Q4).

  * ``extract_targets(text, component_type=0)`` — flat-string path. Sends
    the entire transcript as one input with ``component_type=0`` (an
    undefined CIQ code that flips ``build_extraction_prompt`` into its
    fallback branch) and skips transcript-level dedup. Targets are stamped
    with ``component_type=0``, which mismatches ChromaDB historicals
    (ingested with ctype 2/3/4) and produces systematically worse continuity
    classification.

This node prefers the structured path whenever NB05 hands us a ``transcript``
dict in state. We only drop to the flat path when components are unavailable,
and we log a clear warning in that case.

Configuration (read from environment at extractor instantiation time)
---------------------------------------------------------------------
The LLM extractor is wired through ``LLMTargetExtractor`` and is fully
env-var driven so the same code path runs against OpenAI, Gemini (via
the OpenAI-compatible endpoint), or any other OpenAI-compatible host:

  EARNINGSLENS_LLM_BACKEND   default: "openai"   (uses AsyncOpenAI SDK)
  EARNINGSLENS_LLM_MODEL     default: "gemini-2.5-flash-lite"
  EARNINGSLENS_LLM_BASE_URL  default: Gemini's OpenAI-compatible endpoint
                             "https://generativelanguage.googleapis.com/v1beta/openai/"
  EARNINGSLENS_LLM_API_KEY   primary key var; falls back to GOOGLE_API_KEY,
                             then GEMINI_API_KEY, then OPENAI_API_KEY.

When the resolved ``base_url`` contains "googleapis.com", LLMTargetExtractor
auto-enables its Gemini-compat shim (``_is_gemini_compat=True``), which
appends a JSON-only suffix to the system prompt to suppress chain-of-thought
leakage that breaks strict JSON parsing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy importer helpers
# ---------------------------------------------------------------------------

def _import_llm_extractor():
    """Lazily import LLMTargetExtractor to avoid circular imports."""
    try:
        from src.llm_extraction.llm_extractor import LLMTargetExtractor
        return LLMTargetExtractor
    except ImportError as exc:
        logger.warning("Could not import LLMTargetExtractor: %s", exc)
        return None


def _import_spacy_extractor():
    """Lazily import SpacyTargetExtractor to avoid circular imports."""
    try:
        from src.baseline.target_extractor import SpacyTargetExtractor
        return SpacyTargetExtractor
    except ImportError as exc:
        logger.warning("Could not import SpacyTargetExtractor: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_llm_kwargs() -> Dict[str, Any]:
    """
    Build kwargs for ``LLMTargetExtractor(...)`` from environment variables.

    Defaults target Gemini's OpenAI-compatible endpoint with
    ``gemini-2.5-flash-lite`` because (a) NB05's calibration phase used Google
    keys and (b) flash-lite gives the best $/transcript for the validation
    sweep. Override any of these with the EARNINGSLENS_LLM_* vars without
    touching code.

    Empty strings are treated as "not set" so that an accidentally exported
    blank var doesn't override the default.
    """
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

    kwargs: Dict[str, Any] = {
        "backend": backend,
        "model": model,
        "base_url": base_url,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _structured_transcript_from_state(state: PipelineState) -> Optional[Dict[str, Any]]:
    """
    Return a structured transcript dict suitable for
    ``LLMTargetExtractor.extract_transcript`` if NB05 populated one.

    NB05 should populate ``state['transcript']`` (or ``state['transcript_dict']``)
    with the same shape NB03 v2's ``build_transcript_list`` produces:

        {
          "transcript_id": "<company_id>_<YYYY>Q<N>",
          "company_id"   : "<id>",
          "quarter"      : "<YYYY>Q<N>",
          "components"   : [
              {"text": "...", "component_type": 2},
              {"text": "...", "component_type": 3},
              {"text": "...", "component_type": 4},
              ...
          ],
        }

    Component-type codes follow the CIQ schema:
        1 = Press Release
        2 = Presentation
        3 = Analyst Question
        4 = Management Answer

    A dict is only considered "structured" if it carries at least one
    component with non-empty text and a ctype in {1, 2, 3, 4}. A dict with
    only ctype=0 entries is treated as flat-string data, since ctype=0 is
    NOT a valid CIQ code and would flip the LLM prompt into its fallback
    branch — defeating the purpose of the structured path.
    """
    candidate = state.get("transcript") or state.get("transcript_dict")
    if not isinstance(candidate, dict):
        return None
    components = candidate.get("components")
    if not isinstance(components, list) or not components:
        return None

    valid_ctypes = {1, 2, 3, 4}
    has_valid = False
    for comp in components:
        if not isinstance(comp, dict):
            continue
        text = (comp.get("text") or "").strip()
        if not text:
            continue
        try:
            ctype = int(comp.get("component_type", 0))
        except (TypeError, ValueError):
            continue
        if ctype in valid_ctypes:
            has_valid = True
            break

    return candidate if has_valid else None


# ---------------------------------------------------------------------------
# Extractor node
# ---------------------------------------------------------------------------

async def extractor_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: extract forward-looking targets from the transcript.

    This node is async because ``LLMTargetExtractor.extract_transcript`` /
    ``extract_targets`` are async (they speak to OpenAI / Gemini / a local
    OpenAI-compatible endpoint via ``AsyncOpenAI``). LangGraph supports async
    nodes natively, so the graph invocation site (``graph.ainvoke(state)``)
    does not need to change.

    Execution flow
    --------------
    1. Run SpacyTargetExtractor (always, for baseline comparison).
    2. Attempt LLM extraction:
         - Preferred: ``extract_transcript(state['transcript'])`` when a
           structured transcript dict is available. This bucket-by-component
           path is the one NB03 v2 used to produce ``llm_targets.parquet``,
           which means it reproduces the exact targets that ChromaDB
           historicals were embedded from.
         - Fallback: ``extract_targets(state['transcript_text'],
           component_type=0)`` flat-string path. Used only when no structured
           dict is available; logs a loud warning because the strict-MT score
           may diverge from the NB04 reference.
    3. If LLM extractor succeeds with non-empty results, use those as
       ``extracted_targets`` and save spaCy results in ``spacy_baseline_targets``.
    4. If LLM extractor fails or returns empty results, fall back to spaCy
       results as the primary ``extracted_targets`` and log the fallback.

    Parameters
    ----------
    state : PipelineState
        Must contain at least ``transcript_text`` (string).
        For best results, also populate ``transcript`` (structured dict with
        ``components`` list carrying CIQ component_type codes 2/3/4).

    Returns
    -------
    PipelineState: Updated state with ``extracted_targets``,
                   ``spacy_baseline_targets``, and any new ``errors``.
    """
    transcript_text = state.get("transcript_text", "")
    company_id = state.get("company_id", "UNKNOWN")
    ticker = state.get("ticker", "")
    quarter = state.get("fiscal_quarter", "")
    errors: List[str] = list(state.get("errors", []))

    structured = _structured_transcript_from_state(state)

    # We need *some* text source; require either flat text or a structured
    # transcript with usable components.
    if not transcript_text and not structured:
        msg = f"[{company_id}] extractor_node: no transcript_text and no structured transcript"
        logger.error(msg)
        errors.append(msg)
        return {**state, "extracted_targets": [], "spacy_baseline_targets": [], "errors": errors}

    # Build a flat string for spaCy (and for the LLM fallback path) when
    # only the structured form is available.
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

    # -----------------------------------------------------------------------
    # Step 1: SpaCy baseline extraction (always run)
    # -----------------------------------------------------------------------
    # SpacyTargetExtractor.extract_targets has the simple signature
    # ``(text, component_type=0)`` — it does NOT accept company_id / ticker /
    # fiscal_quarter (that metadata is carried in PipelineState, not on the
    # extractor call). Passing them used to raise TypeError on every transcript
    # and silently skip spaCy entirely. component_type=0 is "transcript-level"
    # since spaCy does not consume the structured prompt.
    spacy_targets: List[Dict[str, Any]] = []
    SpacyExtractorClass = _import_spacy_extractor()

    if SpacyExtractorClass is not None:
        try:
            spacy_extractor = SpacyExtractorClass()
            spacy_targets = spacy_extractor.extract_targets(
                transcript_text, component_type=0
            )
            logger.info(
                "[%s] spaCy extractor: found %d targets",
                company_id,
                len(spacy_targets),
            )
        except Exception as exc:
            msg = f"[{company_id}] SpacyTargetExtractor failed: {exc}"
            logger.warning(msg)
            errors.append(msg)
    else:
        msg = f"[{company_id}] SpacyTargetExtractor not available"
        logger.warning(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # Step 2: LLM extraction (primary)
    # -----------------------------------------------------------------------
    llm_targets: List[Dict[str, Any]] = []
    llm_succeeded = False
    LLMExtractorClass = _import_llm_extractor()

    if LLMExtractorClass is not None:
        llm_kwargs = _resolve_llm_kwargs()
        if not llm_kwargs.get("api_key"):
            # Don't try to instantiate without a key — LLMTargetExtractor will
            # either crash inside AsyncOpenAI or silently 401 every chunk.
            # Surface a single clear error and let spaCy carry the run.
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
                    # Preferred path — matches NB03 v2 production extraction.
                    # extract_transcript buckets components by ctype, runs
                    # one LLM call per bucket with the right prompt, stamps
                    # ctype=2/3/4 on each target, and dedups at transcript
                    # scope. This reproduces the exact targets ChromaDB
                    # historicals were embedded from.
                    n_components = len(structured.get("components", []))
                    ctypes_seen = sorted({
                        int(c.get("component_type", 0))
                        for c in structured.get("components", [])
                        if (c.get("text") or "").strip()
                    })
                    logger.info(
                        "[%s] LLM extract_transcript: %d components, ctypes=%s",
                        company_id,
                        n_components,
                        ctypes_seen,
                    )
                    extract_call = llm_extractor.extract_transcript(structured)
                else:
                    # Fallback path — flat string with ctype=0. This is
                    # known to diverge from the NB04 reference because the
                    # prompt branch and dedup behaviour differ. Log loudly
                    # so NB05 can flag rows produced this way.
                    warn_msg = (
                        f"[{company_id}] LLM extractor falling back to flat "
                        "extract_targets (component_type=0). State["
                        "'transcript'] missing or has no ctype 1/2/3/4 "
                        "components; results may diverge from NB04 baseline."
                    )
                    logger.warning(warn_msg)
                    errors.append(warn_msg)
                    extract_call = llm_extractor.extract_targets(
                        transcript_text, component_type=0
                    )

                if inspect.iscoroutine(extract_call):
                    llm_targets = await extract_call
                else:
                    # Defensive: handle a hypothetical sync extractor too.
                    llm_targets = extract_call

                if llm_targets:
                    llm_succeeded = True
                    logger.info(
                        "[%s] LLM extractor (%s @ %s): found %d targets",
                        company_id,
                        llm_kwargs.get("model"),
                        llm_kwargs.get("base_url"),
                        len(llm_targets),
                    )
                else:
                    # Surface empty-LLM in state.errors so NB05's batch summary
                    # can flag these rows; otherwise they're indistinguishable
                    # from successful runs downstream.
                    empty_msg = f"[{company_id}] LLM extractor returned empty results"
                    logger.warning(empty_msg)
                    errors.append(empty_msg)
            except Exception as exc:
                msg = f"[{company_id}] LLMTargetExtractor failed: {exc}"
                logger.warning(msg)
                errors.append(msg)
    else:
        msg = f"[{company_id}] LLMTargetExtractor not available; using spaCy fallback"
        logger.warning(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # Step 3: Choose primary extraction result
    # -----------------------------------------------------------------------
    if llm_succeeded:
        extracted_targets = llm_targets
        spacy_baseline_targets = spacy_targets
        logger.info(
            "[%s] Using LLM extraction (%d targets). spaCy baseline: %d targets.",
            company_id,
            len(extracted_targets),
            len(spacy_baseline_targets),
        )
    else:
        # Fallback to spaCy
        extracted_targets = spacy_targets
        spacy_baseline_targets = spacy_targets  # same source, no separate baseline
        # Always record the fallback event WITH its target count — losing the
        # count to a substring de-dup on "fallback" makes downstream debugging
        # harder. De-duplication, if any, should happen at the log-aggregation
        # layer, not here.
        msg = f"[{company_id}] Falling back to spaCy extractor ({len(spacy_targets)} targets)"
        logger.warning(msg)
        errors.append(msg)

    return {
        **state,
        "extracted_targets": extracted_targets,
        "spacy_baseline_targets": spacy_baseline_targets,
        "errors": errors,
    }
