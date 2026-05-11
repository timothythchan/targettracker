"""
classifier_agent.py — Classifier Agent node for the EarningsLens LangGraph pipeline.

This agent classifies each dropped target according to:
    1. Type: financial vs. non-financial
    2. Persistence: whether the target appeared in ALL of the prior 12 consecutive
       quarters (persistent) or not (ephemeral).

It then computes a composite ``risk_score`` in [0, 1] using the formula:

    risk_score = 0.5 * (n_dropped / n_total)
               + 0.3 * persistent_drop_ratio
               + 0.2 * nonfinancial_drop_ratio

Higher risk scores indicate a greater likelihood of management obfuscation —
i.e. the company is quietly abandoning targets that were previously emphasised.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .state import PipelineState

logger = logging.getLogger(__name__)

# Non-financial keywords used for heuristic classification when
# the ``is_financial`` field is absent.
_NON_FINANCIAL_KEYWORDS = frozenset(
    [
        "employee", "headcount", "hiring", "retention", "satisfaction",
        "customer", "net promoter", "nps", "engagement", "diversity",
        "sustainability", "carbon", "emission", "esg", "safety", "incident",
        "brand", "culture", "training", "community", "social",
        "market share",  # treat as non-financial for MT purposes
        "pipeline", "backlog",  # operational targets
        "product launch", "innovation", "r&d output",
    ]
)

_FINANCIAL_KEYWORDS = frozenset(
    [
        "revenue", "sales", "earnings", "eps", "ebitda", "ebit",
        "net income", "gross margin", "operating margin", "free cash flow",
        "fcf", "capex", "guidance", "outlook", "forecast",
        "return on", "roe", "roa", "roic", "dividend",
    ]
)

# Number of consecutive prior quarters required to be "persistent"
_PERSISTENCE_WINDOW = 12


def classifier_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: classify dropped targets and compute risk score.

    Execution flow
    --------------
    1. Retrieve dropped target labels from ``continuity_results``.
    2. Resolve full metadata for each dropped target from ``historical_targets``.
    3. Classify each target as financial or non-financial.
    4. Classify each target as persistent or ephemeral using ChromaDB history.
    5. Compute the composite risk_score.
    6. Write ``classification_results`` to state.

    Parameters
    ----------
    state : PipelineState
        Must contain ``continuity_results``, ``historical_targets``,
        ``company_id``, and ``fiscal_quarter``.

    Returns
    -------
    PipelineState: Updated state with ``classification_results``.
    """
    company_id = state.get("company_id", "UNKNOWN")
    quarter = state.get("fiscal_quarter", "")
    continuity: Dict[str, Any] = state.get("continuity_results", {})
    historical: List[Dict[str, Any]] = state.get("historical_targets", [])
    current_targets: List[Dict[str, Any]] = state.get("extracted_targets", [])
    errors: List[str] = list(state.get("errors", []))

    dropped_labels: List[str] = continuity.get("dropped", [])
    all_hist_labels = (
        continuity.get("maintained", [])
        + continuity.get("rephrased", [])
        + dropped_labels
    )
    n_total = len(all_hist_labels)

    logger.info(
        "[%s] classifier_node: classifying %d dropped targets (total_hist=%d)",
        company_id,
        len(dropped_labels),
        n_total,
    )

    if not dropped_labels:
        # Nothing dropped — zero risk
        result = _zero_classification_result(n_total)
        logger.info("[%s] No dropped targets — risk_score=0.0", company_id)
        return {**state, "classification_results": result, "errors": errors}

    # -----------------------------------------------------------------------
    # Build metadata lookup from historical target list
    # -----------------------------------------------------------------------
    hist_meta: Dict[str, Dict[str, Any]] = {}
    for ht in historical:
        label = f"{ht.get('metric_name', '')} ({ht.get('fiscal_quarter', '')})"
        hist_meta[label] = ht

    # -----------------------------------------------------------------------
    # Classify each dropped target
    # -----------------------------------------------------------------------
    dropped_financial: List[Dict[str, Any]] = []
    dropped_non_financial: List[Dict[str, Any]] = []
    persistent_dropped: List[Dict[str, Any]] = []
    ephemeral_dropped: List[Dict[str, Any]] = []

    # Persistence check requires ChromaDB lookup
    store = _get_store()

    for label in dropped_labels:
        meta = hist_meta.get(label, {})
        metric_name = meta.get("metric_name", label.split(" (")[0])
        fq = meta.get("fiscal_quarter", "")

        # --- Financial classification ---
        is_financial = _classify_financial(meta, metric_name)

        # --- Persistence check ---
        is_persistent = _check_persistence(
            store=store,
            company_id=company_id,
            current_quarter=quarter,
            metric_name=metric_name,
            window=_PERSISTENCE_WINDOW,
        )

        record: Dict[str, Any] = {
            "label": label,
            "metric_name": metric_name,
            "last_seen_quarter": fq,
            "is_financial": is_financial,
            "target_type": "financial" if is_financial else "non-financial",
            "is_persistent": is_persistent,
        }

        if is_financial:
            dropped_financial.append(record)
        else:
            dropped_non_financial.append(record)

        if is_persistent:
            persistent_dropped.append(record)
        else:
            ephemeral_dropped.append(record)

    # -----------------------------------------------------------------------
    # Compute risk score
    # -----------------------------------------------------------------------
    n_dropped = len(dropped_labels)
    prop_dropped = n_dropped / n_total if n_total > 0 else 0.0

    persistent_drop_ratio = (
        len(persistent_dropped) / n_dropped if n_dropped > 0 else 0.0
    )
    nonfinancial_drop_ratio = (
        len(dropped_non_financial) / n_dropped if n_dropped > 0 else 0.0
    )

    risk_score = (
        0.5 * prop_dropped
        + 0.3 * persistent_drop_ratio
        + 0.2 * nonfinancial_drop_ratio
    )
    risk_score = float(min(max(risk_score, 0.0), 1.0))

    classification_results: Dict[str, Any] = {
        "dropped_financial": dropped_financial,
        "dropped_non_financial": dropped_non_financial,
        "persistent_dropped": persistent_dropped,
        "ephemeral_dropped": ephemeral_dropped,
        "n_dropped": n_dropped,
        "n_total": n_total,
        "risk_score": risk_score,
        "financial_drop_ratio": 1.0 - nonfinancial_drop_ratio,
        "nonfinancial_drop_ratio": nonfinancial_drop_ratio,
        "persistent_drop_ratio": persistent_drop_ratio,
    }

    logger.info(
        "[%s] risk_score=%.3f (prop_dropped=%.2f, persistent_ratio=%.2f, "
        "nonfinancial_ratio=%.2f)",
        company_id,
        risk_score,
        prop_dropped,
        persistent_drop_ratio,
        nonfinancial_drop_ratio,
    )

    return {**state, "classification_results": classification_results, "errors": errors}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _classify_financial(meta: Dict[str, Any], metric_name: str) -> bool:
    """
    Determine whether a target is financial or non-financial.

    Uses the ``is_financial`` field from metadata if available; otherwise
    applies keyword heuristics on the metric name.

    Parameters
    ----------
    meta:        Target metadata dict.
    metric_name: Metric name string.

    Returns
    -------
    bool: True if the target is financial.
    """
    if "is_financial" in meta:
        return bool(meta["is_financial"])

    name_lower = metric_name.lower()
    if any(kw in name_lower for kw in _FINANCIAL_KEYWORDS):
        return True
    if any(kw in name_lower for kw in _NON_FINANCIAL_KEYWORDS):
        return False
    # Default: assume financial if unclear
    return True


def _check_persistence(
    store: Optional[Any],
    company_id: str,
    current_quarter: str,
    metric_name: str,
    window: int = _PERSISTENCE_WINDOW,
) -> bool:
    """
    Check whether ``metric_name`` appeared in ALL of the prior ``window`` quarters.

    Uses fuzzy matching on metric_name against ChromaDB-stored targets if the
    store is available; otherwise returns False (conservative — not persistent).

    Parameters
    ----------
    store:            TargetVectorStore instance (may be None if unavailable).
    company_id:       Company identifier.
    current_quarter:  Current fiscal quarter string.
    metric_name:      Target metric name to check.
    window:           Number of consecutive prior quarters to check.

    Returns
    -------
    bool: True if the metric appeared in every one of the prior ``window`` quarters.
    """
    if store is None:
        return False

    try:
        from src.rag.vector_store import TargetVectorStore as TVS
        prior_quarters = TVS._get_prior_quarters(current_quarter, window)
    except Exception:
        return False

    if not prior_quarters:
        return False

    for pq in prior_quarters:
        try:
            results = store._collection.get(
                where={
                    "$and": [
                        {"company_id": {"$eq": company_id}},
                        {"fiscal_quarter": {"$eq": pq}},
                    ]
                },
                include=["metadatas"],
            )
            metadatas = results.get("metadatas") or []
            quarter_metric_names = [
                m.get("metric_name", "").lower()
                for m in metadatas
                if m is not None
            ]
            # Fuzzy match: check if any stored metric name contains the target
            name_lower = metric_name.lower()
            found = any(
                name_lower in stored or stored in name_lower
                for stored in quarter_metric_names
            )
            if not found:
                return False
        except Exception as exc:
            logger.debug("Persistence check failed for quarter %s: %s", pq, exc)
            return False

    return True


def _get_store() -> Optional[Any]:
    """
    Attempt to instantiate the TargetVectorStore for persistence lookups.

    Returns
    -------
    TargetVectorStore instance or None if unavailable.
    """
    try:
        from src.rag.vector_store import TargetVectorStore
        return TargetVectorStore()
    except Exception as exc:
        logger.warning("Could not instantiate TargetVectorStore for persistence check: %s", exc)
        return None


def _zero_classification_result(n_total: int) -> Dict[str, Any]:
    """Return a zero-valued classification result."""
    return {
        "dropped_financial": [],
        "dropped_non_financial": [],
        "persistent_dropped": [],
        "ephemeral_dropped": [],
        "n_dropped": 0,
        "n_total": n_total,
        "risk_score": 0.0,
        "financial_drop_ratio": 0.0,
        "nonfinancial_drop_ratio": 0.0,
        "persistent_drop_ratio": 0.0,
    }
