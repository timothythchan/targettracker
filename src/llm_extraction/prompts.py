"""
prompts.py — Prompt templates for LLM-based earnings target extraction.

All prompts use a chain-of-thought structure that guides the model through
explicit reasoning steps before emitting structured JSON output. Few-shot
examples are chosen specifically to surface targets that traditional spaCy
NER pipelines miss (no MONEY/PERCENT entity, rephrased metrics, product-
type targets with implicit management commitments).

PATCH v2 — Fix #2 supplement (2026-05-05)
- Added COMPONENT_TYPE_LABELS / COMPONENT_TYPE_PRIORITY for CIQ schema
- EXTRACTION_PROMPT now takes {component_type_id} + {component_type_label}
- build_extraction_prompt() updated to pass both fields
- CIQ schema (1=Press Release, 2=Presentation, 3=Analyst Q, 4=Mgmt Answer)
"""

# ---------------------------------------------------------------------------
# COMPONENT TYPE METADATA (CIQ Transcripts schema)
# ---------------------------------------------------------------------------
# componenttypeid values from the CIQ ciqtranscriptcomponent table.
# Cohen & Nguyen (2024) work primarily with types 2, 3, 4.

COMPONENT_TYPE_LABELS: dict[int, str] = {
    1: "press release",
    2: "presentation",
    3: "analyst question",
    4: "management answer",
}

# Higher value = more likely to contain quantitative guidance.
# Used in NB03 to pick the most KPI-rich component for prompt-design demos.
COMPONENT_TYPE_PRIORITY: dict[int, int] = {
    2: 4,   # Presentation — densest with guidance
    4: 3,   # Management Answer — KPIs surface in Q&A responses
    3: 2,   # Analyst Question — often references KPIs without committing
    1: 1,   # Press Release — usually duplicates presentation
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior financial analyst specializing in earnings \
call transcript analysis. Your expertise is identifying PERFORMANCE TARGETS: \
any measurable metric that management has previously tracked, currently tracks, \
or implicitly commits to maintaining.

Performance targets extend well beyond standard financial NER. They include:
  • Quantitative metrics with specific numbers (revenue, margins, EPS, etc.)
  • Trend-based commitments ("double-digit growth", "outpacing benchmarks")
  • Operational metrics (subscribers, units shipped, store counts, active users)
  • Implicit commitments embedded in management commentary
  • Relative comparisons that signal a tracked baseline (YoY, QoQ, vs. peers)

You respond ONLY with valid JSON. Never add prose outside the JSON structure. \
If a transcript segment contains no performance targets, return an empty list: []
"""

# ---------------------------------------------------------------------------
# EXTRACTION PROMPT (chain-of-thought, three-stage)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Analyze the following earnings call transcript segment using the three stages \
below. Work through each stage carefully before writing your final JSON output.

--- TRANSCRIPT SEGMENT ---
{transcript_text}
--- END SEGMENT ---

COMPONENT TYPE: {component_type_id} ({component_type_label})
(CIQ schema — 1=press release, 2=presentation, 3=analyst question, 4=management answer)

=== STAGE 1: Identify all discussed business metrics ===
Read the segment and list every business metric mentioned — whether financial \
(revenue, margins, EPS, costs, prices) or operational (subscribers, units, \
stores, devices, users, market share, retention). Include metrics stated \
numerically AND those described only in qualitative/trend terms. Do not skip \
metrics just because they lack an explicit number.

=== STAGE 2: Classify each metric ===
For each metric from Stage 1, decide:
  TRACKED PERFORMANCE TARGET — management provides a specific numerical value, \
a trend direction (e.g., "double-digit growth"), a growth rate (with or without \
exact %), a relative benchmark comparison, or any explicit or implicit commitment \
to maintaining/improving the metric.
  CASUAL MENTION — the metric is referenced incidentally with no numerical \
context and no management commitment (e.g., "as you know, inflation exists").

=== STAGE 3: Output JSON ===
Return a JSON array. Each element is one TRACKED PERFORMANCE TARGET with these \
exact keys:
  "metric_name"      : string — canonical short name (e.g., "cloud revenue growth")
  "raw_text"         : string — verbatim phrase from transcript that anchors this target
  "numerical_value"  : number or null — the explicit numeric value if present
  "trend_direction"  : string or null — "increasing", "decreasing", "stable", or null
  "unit"             : string or null — e.g., "percent", "USD_millions", "subscribers", null
  "temporal_framing" : "backward_looking" | "forward_guidance" | "mixed"
  "is_financial"     : boolean — true if relates to revenue/earnings/margins/costs/prices; \
false if product/operational (subscribers/units/stores/devices/market_share)
  "confidence"       : "high" | "medium" | "low"

Output ONLY the JSON array. No preamble, no commentary.

{few_shot_examples}
"""

# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# ---------------------------------------------------------------------------
# Three examples specifically targeting spaCy pipeline blind spots:
#   1. Trend-based target with no MONEY/PERCENT entity
#   2. Rephrased same-store sales metric spaCy would flag as a drop
#   3. Product-type operational target with implicit management commitment

FEW_SHOT_EXAMPLES = """\
=== FEW-SHOT EXAMPLES (for reference — do not include in your output) ===

--- EXAMPLE 1 ---
Transcript: "We have sustained double-digit growth in our cloud infrastructure \
segment for six consecutive quarters, and we see no reason that trajectory \
changes in the near term."
Correct output:
[
  {{
    "metric_name": "cloud infrastructure segment growth",
    "raw_text": "sustained double-digit growth in our cloud infrastructure segment for six consecutive quarters",
    "numerical_value": null,
    "trend_direction": "increasing",
    "unit": "percent",
    "temporal_framing": "mixed",
    "is_financial": false,
    "confidence": "high"
  }}
]
WHY: No explicit MONEY or PERCENT token — spaCy NER would miss this entirely. \
But "double-digit growth … six consecutive quarters" is a clear tracked target \
with an implicit forward commitment ("we see no reason that trajectory changes").

--- EXAMPLE 2 ---
Transcript: "Comparable-store sales momentum continues to outpace industry \
benchmarks, reflecting the strength of our loyalty program and operational \
discipline across the portfolio."
Correct output:
[
  {{
    "metric_name": "comparable-store sales growth vs. industry",
    "raw_text": "Comparable-store sales momentum continues to outpace industry benchmarks",
    "numerical_value": null,
    "trend_direction": "increasing",
    "unit": "percent",
    "temporal_framing": "backward_looking",
    "is_financial": true,
    "confidence": "high"
  }}
]
WHY: "Comparable-store sales" is a well-known KPI synonym for "same-store sales \
growth." The phrase "outpace industry benchmarks" provides a relative target. \
A naive spaCy pipeline might flag this as a sales *drop* due to missing numeric \
context. This is a tracked financial target with a clear management commitment.

--- EXAMPLE 3 ---
Transcript: "Our installed base now exceeds 2 billion active devices, growing \
at a pace we haven't seen in years, which continues to drive service attachment \
rates across the ecosystem."
Correct output:
[
  {{
    "metric_name": "active device installed base",
    "raw_text": "installed base now exceeds 2 billion active devices, growing at a pace we haven\u2019t seen in years",
    "numerical_value": 2000000000,
    "trend_direction": "increasing",
    "unit": "devices",
    "temporal_framing": "backward_looking",
    "is_financial": false,
    "confidence": "high"
  }},
  {{
    "metric_name": "service attachment rate",
    "raw_text": "continues to drive service attachment rates across the ecosystem",
    "numerical_value": null,
    "trend_direction": "increasing",
    "unit": null,
    "temporal_framing": "backward_looking",
    "is_financial": true,
    "confidence": "medium"
  }}
]
WHY: The device count (2 billion) is a CARDINAL/PRODUCT entity, not a MONEY \
or PERCENT entity — spaCy's financial NER would likely miss it. The implicit \
commitment ("growing at a pace we haven't seen in years") elevates it to a \
tracked target. Service attachment rate is a derived financial metric surfaced \
only through ecosystem language.
=== END FEW-SHOT EXAMPLES ===
"""

# Full extraction prompt with examples embedded
EXTRACTION_PROMPT_WITH_EXAMPLES = EXTRACTION_PROMPT.format(
    transcript_text="{transcript_text}",
    component_type_id="{component_type_id}",
    component_type_label="{component_type_label}",
    few_shot_examples=FEW_SHOT_EXAMPLES,
)

# ---------------------------------------------------------------------------
# CONTINUITY CHECK PROMPT
# ---------------------------------------------------------------------------
# Used as fallback in RAG-based deduplication: given two target descriptions,
# decide whether they refer to the same underlying tracked metric.

CONTINUITY_CHECK_PROMPT = """\
You are a financial data normalisation expert. Determine whether the two \
performance target descriptions below refer to the SAME underlying business \
metric tracked by management, even if worded differently across quarters.

TARGET A:
  metric_name   : {metric_name_a}
  raw_text      : "{raw_text_a}"
  is_financial  : {is_financial_a}
  temporal      : {temporal_a}

TARGET B:
  metric_name   : {metric_name_b}
  raw_text      : "{raw_text_b}"
  is_financial  : {is_financial_b}
  temporal      : {temporal_b}

Respond with a JSON object containing exactly two keys:
  "same_metric" : boolean — true if these describe the same underlying KPI
  "confidence"  : "high" | "medium" | "low"
  "rationale"   : string — one-sentence explanation

Rules:
- Same metric means the SAME business KPI even if labelled differently \
  (e.g., "comp-store sales" and "comparable-store sales growth" → same metric).
- Direction or temporal framing differences alone do not make them different.
- Different segments of the business (cloud vs. hardware) make them different.
- Cross-category differences (financial vs. operational) are usually different.

Output ONLY the JSON object.
"""

# ---------------------------------------------------------------------------
# CANONICALIZATION (cross-pipeline metric-name matching)
# ---------------------------------------------------------------------------
# canonicalize_metric() produces a single string key from BOTH:
#   * spaCy's `normalized_text` (already lemmatized & number-stripped)
#   * LLM's `metric_name` (raw natural language, e.g. "Gross Margin")
#
# Used by NB03 cells 17 and 44, NB04 RAG matching, NB07 fuzzy F1, etc., to
# treat "Gross Margin", "gross margins", "gross-margin" as one canonical
# target. Tokenizer-free so it can run inside groupby().apply() without
# loading spaCy. Output is lower-case, singular-where-trivial, alphanumeric
# words separated by single spaces.

import re as _re

# KPI synonym groups — most common rephrasings analysts/management use
# interchangeably. Maps surface form → canonical key. Order-insensitive
# substring match (whole-word boundary).
_SYNONYM_MAP: dict[str, str] = {
    # Revenue family
    "revenues": "revenue",
    "net sales": "revenue",
    "sales": "revenue",
    "top line": "revenue",
    "top-line": "revenue",
    "topline": "revenue",
    # Margin family
    "gross margins": "gross margin",
    "operating margins": "operating margin",
    "net margins": "net margin",
    "ebitda margins": "ebitda margin",
    # Earnings family
    "earnings per share": "eps",
    "earning per share": "eps",
    # Same-store sales
    "comparable store sales": "comp sales",
    "comparable-store sales": "comp sales",
    "comp store sales": "comp sales",
    "same store sales": "comp sales",
    "same-store sales": "comp sales",
    # Common operational
    "active users": "users",
    "monthly active users": "mau",
    "daily active users": "dau",
    # Cash flow
    "free cash flows": "free cash flow",
    "operating cash flow": "cash flow from operations",
    "cash flow from operating activities": "cash flow from operations",
}

# Filler words to strip — non-semantic determiners and possessives.
_FILLER_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "our", "my", "its", "their", "his", "her",
    "this", "that", "these", "those",
    "of", "in", "on", "for", "to", "from", "as", "by", "at", "with",
    "and", "or",
})

# Trivial plural-stripping suffixes for English KPI vocab.
# More cautious than nltk/spacy lemmatization but works on the common cases.
_PLURAL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("sses", "ss"),
    ("shes", "sh"),
    ("ches", "ch"),
    ("xes", "x"),
    ("s", ""),
)


def _strip_plural(word: str) -> str:
    if len(word) <= 3:  # avoid 'gas' → 'ga', 'cos' → 'co', etc.
        return word
    if word in {"sales", "earnings", "news", "savings", "proceeds",
                "gross", "loss", "basis", "analysis", "hypothesis",
                "miss", "focus", "chassis", "bias", "crisis"}:
        # English mass-singulars / -ss / -is words that look plural
        return word
    if word.endswith("ss"):  # e.g. 'gross', 'progress', 'business'
        return word
    for suf, repl in _PLURAL_SUFFIXES:
        if word.endswith(suf):
            cand = word[: -len(suf)] + repl
            if len(cand) >= 2:
                return cand
    return word


def canonicalize_metric(text: str) -> str:
    """
    Reduce a metric name to a canonical, lower-case, hyphen-free string
    suitable for set-based matching across spaCy and LLM extractions.

    Steps
    -----
    1. Lower-case, strip leading/trailing whitespace.
    2. Apply synonym map (e.g. "comparable-store sales" → "comp sales").
    3. Strip currency symbols, raw numbers, and unit tokens.
    4. Replace hyphens / underscores / slashes with spaces.
    5. Drop punctuation; collapse whitespace.
    6. Tokenize on whitespace; drop filler determiners/prepositions.
    7. Light plural-stripping ("margins" → "margin").
    8. Re-join with single spaces.

    Parameters
    ----------
    text : str
        Raw metric name from spaCy `target_text` / `normalized_text`
        or LLM `metric_name`.

    Returns
    -------
    str
        Canonical key. Empty string if input is empty or yields no
        meaningful tokens.
    """
    if text is None:
        return ""
    s = str(text).lower().strip()
    if not s:
        return ""

    # 2. Synonym substitution (whole-word, longest-match-first).
    # Hyphens normalised first so "comparable-store" matches "comparable store".
    s_for_synonyms = _re.sub(r"[-_/]", " ", s)
    s_for_synonyms = _re.sub(r"\s+", " ", s_for_synonyms)
    for surface, canon in sorted(_SYNONYM_MAP.items(), key=lambda kv: -len(kv[0])):
        s_for_synonyms = _re.sub(
            rf"\b{_re.escape(surface)}\b", canon, s_for_synonyms
        )
    s = s_for_synonyms

    # 3a. Currency symbols
    s = _re.sub(r"[\$\u20ac\u00a3\u00a5]", " ", s)
    # 3b. Numbers (including FY suffixes like 'fy24', 'q1', 'h2', '2024')
    s = _re.sub(r"\b(?:fy|q|h)\d+\b", " ", s)
    s = _re.sub(r"\b\d+(?:[.,]\d+)*\b", " ", s)
    # 3c. Units / magnitudes
    s = _re.sub(
        r"\b(percent(?:age)?|pct|bps|basis\s+points?|bp|"
        r"million|billion|trillion|thousand|"
        r"mn|bn|tn|mm|k|x)\b",
        " ",
        s,
    )
    s = s.replace("%", " ")

    # 4. Hyphens / underscores / slashes
    s = _re.sub(r"[-_/]", " ", s)

    # 5. Punctuation
    s = _re.sub(r"[^a-z0-9\s]", " ", s)

    # 6. Tokenize, drop fillers
    tokens = [t for t in s.split() if t and t not in _FILLER_WORDS]
    if not tokens:
        return ""

    # 7. Plural-strip
    tokens = [_strip_plural(t) for t in tokens]

    return " ".join(tokens).strip()


# ---------------------------------------------------------------------------
# UTILITY: format helpers
# ---------------------------------------------------------------------------

def build_extraction_prompt(transcript_text: str, component_type: int) -> str:
    """
    Return the fully-rendered extraction prompt for a given transcript segment.

    Parameters
    ----------
    transcript_text : str
        Raw text of the transcript segment.
    component_type : int
        CIQ componenttypeid (1=press release, 2=presentation, 3=analyst Q,
        4=management answer).

    Returns
    -------
    str
        Rendered prompt ready to send as the user message.
    """
    label = COMPONENT_TYPE_LABELS.get(int(component_type), "other")
    return EXTRACTION_PROMPT_WITH_EXAMPLES.format(
        transcript_text=transcript_text,
        component_type_id=component_type,
        component_type_label=label,
    )


def build_continuity_check_prompt(
    metric_name_a: str,
    raw_text_a: str,
    is_financial_a: bool,
    temporal_a: str,
    metric_name_b: str,
    raw_text_b: str,
    is_financial_b: bool,
    temporal_b: str,
) -> str:
    """
    Return the fully-rendered continuity-check prompt for two target descriptions.
    """
    return CONTINUITY_CHECK_PROMPT.format(
        metric_name_a=metric_name_a,
        raw_text_a=raw_text_a,
        is_financial_a=is_financial_a,
        temporal_a=temporal_a,
        metric_name_b=metric_name_b,
        raw_text_b=raw_text_b,
        is_financial_b=is_financial_b,
        temporal_b=temporal_b,
    )
