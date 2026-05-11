"""
earningslens.src.baseline
=========================
Layer 1 — spaCy Baseline Pipeline

Implements the Cohen & Nguyen (2024) "Moving Targets" NLP extraction
methodology using spaCy NER and dependency parsing.

Public API
----------
SpacyTargetExtractor  — extract performance targets from transcript text
MovingTargetsComputer — compute MT_{i,t} and related measures
BaselinePipeline      — end-to-end orchestration
add_persistence_flags — convenience helper to augment MT DataFrame
"""

from .target_extractor import SpacyTargetExtractor, GOAL_VERBS, NER_LABELS, FINANCIAL_LABELS
from .moving_targets import MovingTargetsComputer, add_persistence_flags
from .baseline_pipeline import BaselinePipeline

__all__ = [
    "SpacyTargetExtractor",
    "MovingTargetsComputer",
    "BaselinePipeline",
    "add_persistence_flags",
    "GOAL_VERBS",
    "NER_LABELS",
    "FINANCIAL_LABELS",
]
