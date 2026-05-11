"""
earningslens.src.llm_extraction
================================

LLM-powered semantic performance-target extraction module for EarningsLens.

This package replaces the spaCy NER pipeline with a chain-of-thought LLM
approach that captures targets the NER pipeline misses — trend-based
commitments, rephrased KPIs, and operational/product metrics with implicit
management commitments.

Modules
-------
prompts
    All prompt templates (system prompt, extraction prompt, few-shot examples,
    continuity-check prompt) and format-helper functions.

llm_extractor
    ``LLMTargetExtractor`` — async LLM extraction engine supporting the
    OpenAI API (GPT-4o / GPT-4o-mini) and local vLLM/Ollama servers
    (Mistral 7B). Includes chunking, deduplication, retry logic, and
    automatic spaCy fallback.

fine_tuning
    QLoRA fine-tuning utilities for Mistral 7B: data preparation,
    LoRA configuration, HuggingFace Trainer loop, and vLLM export.

extraction_pipeline
    End-to-end orchestration pipeline: load transcripts → LLM extraction →
    save Parquet → compare vs. spaCy baseline → log summary statistics.
    Includes CLI entry point.

Quick start
-----------
>>> import asyncio
>>> from earningslens.src.llm_extraction import LLMTargetExtractor
>>>
>>> extractor = LLMTargetExtractor(backend="openai", model="gpt-4o-mini")
>>> targets = asyncio.run(
...     extractor.extract_targets(
...         "We sustained double-digit growth for six consecutive quarters.",
...         component_type=0,
...     )
... )
>>> print(targets)
"""

from .llm_extractor import LLMTargetExtractor
from .extraction_pipeline import PipelineConfig, run_pipeline
from .prompts import (
    SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    FEW_SHOT_EXAMPLES,
    CONTINUITY_CHECK_PROMPT,
    build_extraction_prompt,
    build_continuity_check_prompt,
)

__all__ = [
    # Core extractor
    "LLMTargetExtractor",
    # Pipeline
    "PipelineConfig",
    "run_pipeline",
    # Prompts
    "SYSTEM_PROMPT",
    "EXTRACTION_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "CONTINUITY_CHECK_PROMPT",
    "build_extraction_prompt",
    "build_continuity_check_prompt",
]
