"""EarningsLens RAG module — ChromaDB vector store and semantic continuity matching."""

from .vector_store import TargetVectorStore, DEFAULT_THRESHOLDS
from .semantic_matcher import SemanticContinuityMatcher

__all__ = ["TargetVectorStore", "DEFAULT_THRESHOLDS", "SemanticContinuityMatcher"]
