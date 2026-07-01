"""
graph.py — LangGraph Pipeline Assembly for EarningsLens.

Assembles the full multi-agent StateGraph and exposes convenience functions
for single-transcript and batch processing.

Graph topology
--------------
    extractor → comparator → classifier → reporter → END

Conditional edges
-----------------
When extraction returns no targets, the pipeline still proceeds to the
comparator, which handles empty input gracefully. The reporter can then
produce a minimal "insufficient data" report instead of hard-failing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .state import PipelineState, make_initial_state
from .extractor_agent import extractor_node
from .comparator_agent import comparator_node
from .classifier_agent import classifier_node
from .reporter_agent import reporter_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph imports (lazy to fail gracefully if not installed)
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END as LANGGRAPH_END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning(
        "langgraph not installed. Graph compilation will fail. "
        "Install with: pip install langgraph"
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph: The compiled LangGraph application ready for invocation.

    Raises
    ------
    ImportError: If langgraph is not installed.
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is required for graph compilation. "
            "Install it with: pip install langgraph"
        )

    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("extractor", extractor_node)
    graph.add_node("comparator", comparator_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("reporter", reporter_node)

    # Set entry point
    graph.set_entry_point("extractor")

    # Linear edges
    graph.add_edge("extractor", "comparator")
    graph.add_edge("comparator", "classifier")
    graph.add_edge("classifier", "reporter")
    graph.add_edge("reporter", LANGGRAPH_END)

    # Compile and return
    app = graph.compile()
    logger.info("EarningsLens LangGraph pipeline compiled successfully")
    return app


# ---------------------------------------------------------------------------
# Module-level compiled app (lazy singleton)
# ---------------------------------------------------------------------------
_compiled_app: Optional[Any] = None


def _get_app() -> Any:
    """
    Return the module-level compiled LangGraph app, building it if necessary.

    Returns
    -------
    CompiledGraph: Singleton compiled application.
    """
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_graph()
    return _compiled_app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    transcript_text: str,
    company_id: str,
    ticker: str,
    quarter: str,
    component_type: int = 3,
    use_prebuilt_app: bool = True,
) -> Dict[str, Any]:
    """
    Run the full EarningsLens pipeline on a single transcript.

    Parameters
    ----------
    transcript_text:
        Raw earnings call transcript text.
    company_id:
        Unique company identifier (CIK or ticker).
    ticker:
        Stock ticker symbol (e.g. "AAPL").
    quarter:
        Fiscal quarter string (e.g. "2023Q2").
    component_type:
        Transcript component flag (1=prepared remarks, 2=Q&A, 3=full).
    use_prebuilt_app:
        If True, use the module-level singleton compiled app. Set to False to
        force a fresh graph build (useful in testing).

    Returns
    -------
    Dict[str, Any]: Final pipeline state as a plain dict. Key fields:
        extracted_targets, continuity_results, classification_results, report.
    """
    initial_state = make_initial_state(
        transcript_text=transcript_text,
        company_id=company_id,
        ticker=ticker,
        fiscal_quarter=quarter,
        component_type=component_type,
    )

    start_t = time.perf_counter()
    logger.info(
        "run_pipeline: starting for %s %s (transcript_len=%d)",
        company_id,
        quarter,
        len(transcript_text),
    )

    try:
        app = _get_app() if use_prebuilt_app else build_graph()
        final_state: Dict[str, Any] = app.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline execution failed for %s %s: %s", company_id, quarter, exc)
        # Return a partial state with error information
        final_state = dict(initial_state)
        final_state.setdefault("errors", []).append(f"Pipeline execution failed: {exc}")
        final_state.setdefault("report", {})["summary"] = (
            f"Pipeline execution failed: {exc}"
        )

    elapsed = time.perf_counter() - start_t
    logger.info(
        "run_pipeline: completed for %s %s in %.2fs", company_id, quarter, elapsed
    )

    return final_state


def run_batch(
    transcripts: List[Dict[str, Any]],
    use_prebuilt_app: bool = True,
    stop_on_error: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the EarningsLens pipeline on multiple transcripts sequentially.

    Parameters
    ----------
    transcripts:
        List of transcript dicts. Each must contain:
            transcript_text : str
            company_id      : str
            ticker          : str
            quarter         : str  (or fiscal_quarter)
        Optional:
            component_type  : int (defaults to 3)
    use_prebuilt_app:
        If True, reuse the same compiled app across all transcripts (faster).
    stop_on_error:
        If True, raise on first pipeline error. If False (default), log the
        error and continue with remaining transcripts.

    Returns
    -------
    List[Dict[str, Any]]: List of final pipeline states, one per input transcript.
                          Failed pipelines include ``errors`` in their state.
    """
    results: List[Dict[str, Any]] = []
    total = len(transcripts)

    # Pre-build app once if reusing
    if use_prebuilt_app and _LANGGRAPH_AVAILABLE:
        app = _get_app()
    else:
        app = None  # will build per-call

    for i, t in enumerate(transcripts, 1):
        company_id = t.get("company_id", "UNKNOWN")
        quarter = t.get("quarter") or t.get("fiscal_quarter", "")
        ticker = t.get("ticker", company_id)

        logger.info("run_batch: processing %d/%d — %s %s", i, total, company_id, quarter)

        try:
            result = run_pipeline(
                transcript_text=t.get("transcript_text", ""),
                company_id=company_id,
                ticker=ticker,
                quarter=quarter,
                component_type=t.get("component_type", 3),
                use_prebuilt_app=(app is not None),
            )
            results.append(result)
        except Exception as exc:
            msg = f"run_batch: pipeline failed for {company_id} {quarter}: {exc}"
            logger.error(msg)
            if stop_on_error:
                raise
            error_result: Dict[str, Any] = {
                "company_id": company_id,
                "ticker": ticker,
                "fiscal_quarter": quarter,
                "errors": [msg],
                "report": {"summary": msg, "risk_flag": "LOW", "dropped_targets": []},
            }
            results.append(error_result)

    logger.info("run_batch: completed %d/%d transcripts", len(results), total)
    return results


# ---------------------------------------------------------------------------
# Convenience: expose compiled app at module level
# ---------------------------------------------------------------------------
def get_compiled_app() -> Any:
    """
    Return the compiled LangGraph application.

    This is the primary entrypoint for direct LangGraph invocation:

    >>> app = get_compiled_app()
    >>> result = app.invoke(initial_state)

    Returns
    -------
    CompiledGraph: The singleton compiled EarningsLens pipeline.
    """
    return _get_app()


# Build the app at import time so it is ready on first call.
# Wrap in try/except so importing this module doesn't crash if langgraph
# is not installed.
try:
    app = build_graph()
except Exception as _build_exc:
    app = None  # type: ignore[assignment]
    logger.warning("Could not build LangGraph app at import time: %s", _build_exc)
