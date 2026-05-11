"""
earningslens.src.llm_extraction
================================
LLM-powered semantic performance-target extraction module for EarningsLens.

Imports are resolved lazily so the package remains importable before optional
LLM client dependencies are installed.
"""

__all__ = [
    "LLMTargetExtractor",
    "PipelineConfig",
    "run_pipeline",
    "SYSTEM_PROMPT",
    "EXTRACTION_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "CONTINUITY_CHECK_PROMPT",
    "build_extraction_prompt",
    "build_continuity_check_prompt",
]


def __getattr__(name: str):
    if name == "LLMTargetExtractor":
        from .llm_extractor import LLMTargetExtractor

        return LLMTargetExtractor
    if name in {"PipelineConfig", "run_pipeline"}:
        from . import extraction_pipeline

        return getattr(extraction_pipeline, name)
    if name in {
        "SYSTEM_PROMPT",
        "EXTRACTION_PROMPT",
        "FEW_SHOT_EXAMPLES",
        "CONTINUITY_CHECK_PROMPT",
        "build_extraction_prompt",
        "build_continuity_check_prompt",
    }:
        from . import prompts

        return getattr(prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
