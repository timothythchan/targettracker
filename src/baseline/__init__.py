"""
earningslens.src.baseline
=========================
Layer 1 — spaCy Baseline Pipeline.

Imports are resolved lazily so command-line entry points can be discovered
without importing heavyweight NLP dependencies until the relevant class or
constant is used.
"""

__all__ = [
    "SpacyTargetExtractor",
    "MovingTargetsComputer",
    "BaselinePipeline",
    "add_persistence_flags",
    "GOAL_VERBS",
    "NER_LABELS",
    "FINANCIAL_LABELS",
]


def __getattr__(name: str):
    if name in {"SpacyTargetExtractor", "GOAL_VERBS", "NER_LABELS", "FINANCIAL_LABELS"}:
        from . import target_extractor

        return getattr(target_extractor, name)
    if name in {"MovingTargetsComputer", "add_persistence_flags"}:
        from . import moving_targets

        return getattr(moving_targets, name)
    if name == "BaselinePipeline":
        from .baseline_pipeline import BaselinePipeline

        return BaselinePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
