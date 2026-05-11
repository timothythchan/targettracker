"""
comparator_agent.py — Comparator Agent node for the EarningsLens LangGraph pipeline.

This agent queries the ChromaDB vector store for historical targets at the
paper-strict t-4 lag (single quarter, exactly four quarters before the current
filing), computes embedding-based cosine similarity between the current and
historical targets, and classifies each historical target as maintained,
rephrased, or dropped.

The continuity results are written to ``state.continuity_results`` and the raw
historical target records to ``state.historical_targets`` for downstream use by
the Classifier and Reporter agents.

Notes
-----
v2 (2026-05-07): Aligned with NB04 batch methodology.
    * Thresholds locked to manual midpoints derived from the t-4 strict
      distribution: maintained >= 0.65, 0.40 <= rephrased < 0.65, dropped < 0.40.
      Replaces the legacy 0.80/0.55 union-window thresholds.
    * Historical retrieval switched to ``lag_only=True`` so the comparator
      returns only the single t-4 quarter, matching ``compute_semantic_mt``
      with ``historical_lag=4`` used to build ``semantic_mt_scores.parquet``.
    * Provenance: each continuity result now records ``historical_lag`` so
      downstream consumers can verify retrieval mode at runtime.

v2.1 (2026-05-07): ChromaDB location made explicit.
    * vector_store._DEFAULT_PERSIST_DIR resolves wrong on Colab
      (Path(__file__).parents[3] lands on MyDrive, not EarningsLens).
    * Honour EARNINGSLENS_CHROMA_DIR and EARNINGSLENS_CHROMA_COLLECTION
      env vars so notebooks can point the agent at the populated
      experiment store without monkeypatching vector_store.py.
    * Empty-collection guard: log a loud warning and append to state.errors
      when the resolved collection has zero documents (prevents silent
      all-dropped demo runs).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — kept in sync with NB04 batch (semantic_mt_scores.meta.json)
# ---------------------------------------------------------------------------

# Paper-strict t-4 thresholds (manual midpoints; calibration deferred to v2).
# Source of truth lives in semantic_mt_scores.meta.json -> retrieval_mode=t-4.
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "maintained": 0.50,
    "rephrased": 0.45,
}

# Single-quarter t-4 lag (paper-strict). The previous behaviour queried the
# union of t-1..t-4 which over-counts historicals ~4x and inflates continuity.
_HISTORICAL_LAG: int = 4

# Optional environment overrides for the ChromaDB location and collection.
# Set these in NB05 / NB08 setup cells so the agent points at the populated
# experiment store instead of the (broken) computed default in vector_store.py.
#
#   import os
#   os.environ['EARNINGSLENS_CHROMA_DIR']        = '/content/drive/MyDrive/EarningsLens/data/cache/chromadb_experiment'
#   os.environ['EARNINGSLENS_CHROMA_COLLECTION'] = 'earnings_targets'
#
# When unset, TargetVectorStore() is constructed with no kwargs (legacy path).
_ENV_CHROMA_DIR: str = "EARNINGSLENS_CHROMA_DIR"
_ENV_CHROMA_COLLECTION: str = "EARNINGSLENS_CHROMA_COLLECTION"

_STORE_CACHE: Dict[tuple, "TargetVectorStore"] = {}


def _get_or_create_store(store_kwargs: Dict[str, Any]) -> "TargetVectorStore":
    """Return a cached TargetVectorStore for these kwargs, creating if needed.

    The cache key is (persist_dir, collection_name). Other kwargs (embedding
    model, etc.) are assumed identical for a given (dir, name) pair across
    a notebook run.
    """
    from src.rag.vector_store import TargetVectorStore

    key = (
        store_kwargs.get("persist_dir", ""),
        store_kwargs.get("collection_name", ""),
    )
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached

    store = TargetVectorStore(**store_kwargs)
    _STORE_CACHE[key] = store
    logger.info(
        "Cached new TargetVectorStore: dir=%s collection=%s",
        key[0], key[1],
    )
    return store


def comparator_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: compare current targets against t-4 historical embeddings.

    Execution flow
    --------------
    1. Import TargetVectorStore (lazy) to avoid hard dependency at import time.
    2. Index current extracted_targets into ChromaDB for future lookups.
    3. Query historical targets at the paper-strict t-4 lag (lag_only=True).
    4. Compute pairwise cosine similarity matrix (current x historical).
    5. Classify each historical target as maintained / rephrased / dropped
       using the locked 0.65 / 0.40 thresholds.
    6. Write results to state, including ``historical_lag`` provenance.

    Parameters
    ----------
    state : PipelineState
        Must contain ``extracted_targets``, ``company_id``,
        and ``fiscal_quarter``.

    Returns
    -------
    PipelineState: Updated state with ``historical_targets`` and
                   ``continuity_results``.
    """
    company_id = state.get("company_id", "UNKNOWN")
    quarter = state.get("fiscal_quarter", "")
    current_targets: List[Dict[str, Any]] = state.get("extracted_targets", [])
    errors: List[str] = list(state.get("errors", []))

    logger.info(
        "[%s] comparator_node: comparing %d current targets for %s (lag=t-%d)",
        company_id,
        len(current_targets),
        quarter,
        _HISTORICAL_LAG,
    )

    # -----------------------------------------------------------------------
    # Lazy import of vector store
    # -----------------------------------------------------------------------
    try:
        from src.rag.vector_store import TargetVectorStore
    except ImportError as exc:
        msg = f"[{company_id}] comparator_node: could not import TargetVectorStore: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "historical_targets": [],
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }

    # -----------------------------------------------------------------------
    # Initialise vector store — honour env-var overrides for persist_dir /
    # collection_name so notebooks can point at the populated experiment store.
    # -----------------------------------------------------------------------
    store_kwargs: Dict[str, Any] = {}
    if os.environ.get(_ENV_CHROMA_DIR):
        store_kwargs["persist_dir"] = os.environ[_ENV_CHROMA_DIR]
    if os.environ.get(_ENV_CHROMA_COLLECTION):
        store_kwargs["collection_name"] = os.environ[_ENV_CHROMA_COLLECTION]

    try:
        store = _get_or_create_store(store_kwargs)
    except Exception as exc:
        msg = f"[{company_id}] comparator_node: TargetVectorStore init failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "historical_targets": [],
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }

    # Sanity: a freshly-created empty collection is the #1 silent failure mode.
    # Surface it loudly the first time we see it (count==0) so the demo doesn't
    # quietly produce all-dropped continuity results.
    try:
        n_docs = store._collection.count()
        if n_docs == 0:
            msg = (
                f"[{company_id}] comparator_node: ChromaDB collection "
                f"'{store.collection_name}' at '{store.persist_dir}' is empty. "
                f"Set EARNINGSLENS_CHROMA_DIR / EARNINGSLENS_CHROMA_COLLECTION "
                f"to point at the populated experiment store before running."
            )
            logger.warning(msg)
            errors.append(msg)
    except Exception:
        # Non-fatal — count() is just a sanity check.
        pass

    # -----------------------------------------------------------------------
    # Index current targets into ChromaDB (so future quarters can look them up)
    # -----------------------------------------------------------------------
    if current_targets:
        try:
            n_indexed = store.index_targets(company_id, quarter, current_targets)
            logger.debug("[%s] Indexed %d targets into ChromaDB", company_id, n_indexed)
        except Exception as exc:
            msg = f"[{company_id}] comparator_node: indexing failed: {exc}"
            logger.warning(msg)
            errors.append(msg)

    # -----------------------------------------------------------------------
    # Query historical targets — paper-strict t-4 lag only
    # -----------------------------------------------------------------------
    try:
        historical_targets = store.query_historical_targets(
            company_id=company_id,
            current_quarter=quarter,
            n_quarters=_HISTORICAL_LAG,
            lag_only=True,
        )
    except TypeError as exc:
        # Backward-compat guard: if a stale TargetVectorStore is loaded that
        # does not yet support ``lag_only``, surface a clear, actionable error
        # rather than silently falling back to the union-window behaviour.
        msg = (
            f"[{company_id}] comparator_node: TargetVectorStore.query_historical_targets "
            f"does not accept lag_only=True. Update src/rag/vector_store.py to the v2 "
            f"patched version. Underlying error: {exc}"
        )
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "historical_targets": [],
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }
    except Exception as exc:
        msg = f"[{company_id}] comparator_node: historical query failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "historical_targets": [],
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }

    logger.info(
        "[%s] Retrieved %d historical targets at t-%d",
        company_id,
        len(historical_targets),
        _HISTORICAL_LAG,
    )

    if not historical_targets:
        logger.warning(
            "[%s] No t-%d historical targets found for %s; skipping similarity computation",
            company_id,
            _HISTORICAL_LAG,
            quarter,
        )
        return {
            **state,
            "historical_targets": [],
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }

    # -----------------------------------------------------------------------
    # Compute similarity matrix
    # -----------------------------------------------------------------------
    try:
        sim_matrix = store.compute_similarity(current_targets, historical_targets)
    except Exception as exc:
        msg = f"[{company_id}] comparator_node: similarity computation failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {
            **state,
            "historical_targets": historical_targets,
            "continuity_results": _empty_continuity(),
            "errors": errors,
        }

    # -----------------------------------------------------------------------
    # Classify continuity using the locked 0.65 / 0.40 thresholds
    # -----------------------------------------------------------------------
    try:
        continuity = store.classify_continuity(sim_matrix, _DEFAULT_THRESHOLDS)
    except Exception as exc:
        msg = f"[{company_id}] comparator_node: continuity classification failed: {exc}"
        logger.error(msg)
        errors.append(msg)
        continuity = _empty_continuity()

    # Stamp provenance so downstream agents (Classifier / Reporter) and the
    # demo notebook can verify the retrieval mode without re-reading config.
    continuity = dict(continuity)
    continuity["historical_lag"] = _HISTORICAL_LAG
    continuity["thresholds"] = dict(_DEFAULT_THRESHOLDS)

    logger.info(
        "[%s] Continuity (t-%d, thr=%.2f/%.2f): maintained=%d, rephrased=%d, dropped=%d",
        company_id,
        _HISTORICAL_LAG,
        _DEFAULT_THRESHOLDS["maintained"],
        _DEFAULT_THRESHOLDS["rephrased"],
        len(continuity.get("maintained", [])),
        len(continuity.get("rephrased", [])),
        len(continuity.get("dropped", [])),
    )

    return {
        **state,
        "historical_targets": historical_targets,
        "continuity_results": continuity,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_continuity() -> Dict[str, Any]:
    """Return an empty continuity result structure (with provenance stamps)."""
    return {
        "maintained": [],
        "rephrased": [],
        "dropped": [],
        "details": {},
        "historical_lag": _HISTORICAL_LAG,
        "thresholds": dict(_DEFAULT_THRESHOLDS),
    }
