# EarningsLens — Project Execution Guidebook

**Course:** STAT GR5293 — Generative AI (Columbia University)  
**Team:** Timothy Chan (tc3460) · Yewen Li (yl5888) · Tiantian Hang (th3166)  
**Project:** Replication and LLM Extension of Cohen & Nguyen (2024) "Moving Targets"  
**Last Updated:** March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Week-by-Week Execution Plan](#3-week-by-week-execution-plan)
4. [Module-by-Module User Guide](#4-module-by-module-user-guide)
5. [Data Dictionary](#5-data-dictionary)
6. [Evaluation Interpretation Guide](#6-evaluation-interpretation-guide)
7. [Rubric Alignment Checklist](#7-rubric-alignment-checklist)
8. [Task Division Recommendation](#8-task-division-recommendation)
9. [Troubleshooting FAQ](#9-troubleshooting-faq)
10. [Quick Reference Command Sheet](#10-quick-reference-command-sheet)

---

## 1. Project Overview

### 1.1 Research Question

Do corporate managers revise their performance targets between earnings calls, and does the degree of target revision predict future stock returns? Cohen & Nguyen (2024) in *"Moving Targets: What Do Managers' Performance Targets Tell Us About Future Returns?"* show that measuring how much a manager's stated performance targets shift quarter-over-quarter — the **Moving Targets (MT) measure** — is a powerful predictor of cross-sectional stock returns. Firms with rising targets earn higher subsequent returns; firms with falling targets underperform. Their key insight is that markets are slow to incorporate this forward-looking managerial guidance.

EarningsLens replicates their paper and then extends it using modern LLM techniques to ask: *Can LLMs extract richer, more accurate performance targets than the original rule-based NLP pipeline, and does the improvement in extraction quality generate a stronger return-predictive signal?*

### 1.2 Connection to the Paper

Cohen & Nguyen (2024) use a hand-crafted NLP pipeline (regex + pattern matching) to extract numeric performance targets from earnings call transcripts. They compute MT as the semantic and quantitative shift in these targets quarter-over-quarter, then run Fama-MacBeth cross-sectional regressions showing that high-MT portfolios earn approximately 0.4–0.6% monthly alpha over low-MT portfolios.

EarningsLens:
- **Replicates** their approach using a spaCy-based named entity recognition pipeline (Layer 1)
- **Extends** their approach using GPT-4o chain-of-thought extraction and RAG-based cross-quarter comparison (Layer 2)
- **Evaluates** whether the LLM signal is economically superior via portfolio alphas and Fama-MacBeth regressions (Layer 3)

### 1.3 The Three Layers Explained Simply

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — BASELINE (spaCy)                                 │
│  Input: Raw earnings call transcripts                       │
│  Process: Rule-based NER → extract numeric targets          │
│  Output: spacy_targets.parquet, spacy_mt_scores.parquet     │
│  Purpose: Replication of Cohen & Nguyen                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — LLM EXTENSION                                    │
│  Input: Same transcripts + spaCy outputs as context         │
│  Process: Chain-of-thought GPT-4o extraction                │
│           ChromaDB RAG for cross-quarter matching           │
│           LangGraph 4-agent pipeline                        │
│  Output: llm_targets.parquet, llm_mt_scores.parquet         │
│  Purpose: Richer, more nuanced target extraction            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — EVALUATION                                       │
│  Input: Both sets of MT scores + CRSP/Compustat/IBES data   │
│  Process: Fama-MacBeth regressions, portfolio sorts         │
│           Ablation studies, NLP quality metrics             │
│  Output: Regression tables, alpha estimates, F1 scores      │
│  Purpose: Quantify improvement from LLM extension           │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1 (Baseline)** is a faithful replication of the original paper. The `SpacyTargetExtractor` applies linguistic patterns (dependency parsing + entity rules) to identify sentences where a manager states a future performance metric — e.g., "We expect revenue to reach $2.5 billion next quarter." The `MovingTargetsComputer` then aligns these targets across consecutive quarters and computes the MT score as a normalized directional change.

**Layer 2 (LLM Extension)** uses GPT-4o with structured chain-of-thought prompting to extract targets with richer semantic context (target type, hedging language, confidence level, explicit vs. implied targets). ChromaDB stores target embeddings so that cross-quarter comparison becomes a semantic similarity search rather than exact string matching. The LangGraph pipeline orchestrates four agents: an Extractor, a Comparator (cross-quarter matching), a Classifier (revision direction), and a Reporter (final MT score assembly).

**Layer 3 (Evaluation)** runs the empirical finance tests. Fama-MacBeth regressions test whether MT predicts the next month's returns, controlling for standard factors (Size, BM, Momentum, SUE). Calendar-time portfolios sort firms into quintiles by MT score and report long-minus-short alpha. Ablation studies decompose the LLM's incremental value.

### 1.4 Course Module Mapping

| Course Module | EarningsLens Component |
|---|---|
| Prompt Engineering | `src/llm_extraction/prompts.py` — chain-of-thought templates |
| RAG & Vector Databases | `src/rag/` — ChromaDB indexing + semantic matching |
| Agent Frameworks | `src/agents/` — LangGraph 4-agent pipeline |
| Fine-Tuning (QLoRA) | `src/llm_extraction/fine_tuning.py` — Mistral 7B fine-tuning |
| LLM Evaluation | `src/evaluation/comparison.py` — NLP quality metrics |
| NLP Foundations | `src/baseline/` — spaCy NER pipeline |
| Financial ML | `src/evaluation/fama_macbeth.py` — cross-sectional regressions |
| Demo / Deployment | `demo/app.py` — Gradio interactive demo |

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.10 or higher (3.11 recommended)
- Git
- Columbia WRDS account (verify at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu))
- OpenAI API key with GPT-4o access
- At least 16 GB RAM (32 GB recommended for full dataset)
- At least 20 GB free disk space for raw data

### 2.2 Clone and Install

```bash
# 1. Clone the repository
git clone https://github.com/[your-repo]/earningslens.git
cd earningslens

# 2. Create a Python 3.10+ virtual environment
python3.10 -m venv .venv

# 3. Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip to avoid dependency resolution issues
pip install --upgrade pip setuptools wheel

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Install the earningslens package in editable mode
# (this makes `src` importable from anywhere in the repo)
pip install -e .

# 7. Download the spaCy large English model
python -m spacy download en_core_web_lg

# 8. Verify spaCy installation
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('spaCy OK:', nlp.meta['name'])"
# Expected output: spaCy OK: en_core_web_lg
```

### 2.3 Environment Variables (.env File)

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in your editor and set the following values:

```dotenv
# ── WRDS ────────────────────────────────────────────────────────────
# Your Columbia WRDS username (UNI typically works)
WRDS_USERNAME=tc3460

# WRDS password — set this or use interactive prompt
# Leave blank to be prompted each session (more secure)
WRDS_PASSWORD=

# ── OpenAI ──────────────────────────────────────────────────────────
# Your OpenAI API key — never commit this to Git
OPENAI_API_KEY=sk-...

# Model to use for extraction (gpt-4o-mini is cheaper for testing)
OPENAI_MODEL=gpt-4o-mini

# ── Data Paths ───────────────────────────────────────────────────────
DATA_DIR=./data
RAW_DIR=./data/raw
PROCESSED_DIR=./data/processed
CACHE_DIR=./data/cache

# ── ChromaDB ────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./data/cache/chromadb

# ── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_DIR=./logs

# ── Fine-Tuning (Colab only) ─────────────────────────────────────────
HF_TOKEN=hf_...
```

> **Security note:** `.env` is in `.gitignore`. Never commit API keys. Each team member must set up their own `.env` with their own WRDS credentials and share a single team OpenAI API key to control costs.

### 2.4 WRDS Account Verification

Before running any data pipeline, verify your WRDS connection:

```bash
python - <<'EOF'
import wrds
db = wrds.Connection()   # Will prompt for password if WRDS_PASSWORD is blank
result = db.raw_sql("SELECT COUNT(*) FROM ciq.wrds_transcript_detail LIMIT 1")
print("WRDS connection OK. Row count test:", result)
db.close()
EOF
```

**Expected output:** A successful connection message and a count result.

**If this fails:**
- Ensure your UNI has WRDS access at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu) (sign in via Columbia SSO)
- Request access to these specific WRDS datasets if not already approved:
  - Capital IQ Transcripts (`ciq`)
  - CRSP (`crsp`)
  - Compustat (`comp`)
  - IBES (`ibes`)
  - Fama-French Factors (`ff`)
- WRDS access requests typically take 1–2 business days; submit early

### 2.5 Verify the Full Installation

Run the test suite to confirm everything is working before touching real data:

```bash
pytest tests/test_baseline.py -v
```

**Expected output:** All 39 tests pass. If any fail, check the error message — most are import errors solvable by reinstalling requirements.

Run a quick smoke test across all modules:

```bash
python - <<'EOF'
# Test core imports
import spacy
import chromadb
import langchain
import langgraph
import openai
import wrds
import pandas as pd
import statsmodels.api as sm

# Test spaCy model
nlp = spacy.load("en_core_web_lg")

# Test ChromaDB
client = chromadb.Client()
col = client.create_collection("test")
col.add(documents=["hello world"], ids=["1"])
results = col.query(query_texts=["hello"], n_results=1)
assert results["documents"][0][0] == "hello world"

# Test config loading
from src.utils.config_loader import load_config
cfg = load_config()

print("All imports and smoke tests passed.")
EOF
```

### 2.6 Google Colab Pro+ Setup (Fine-Tuning Only)

The QLoRA fine-tuning step (`src/llm_extraction/fine_tuning.py`) requires a GPU with at least 16 GB VRAM (A100 recommended). Use Google Colab Pro+:

1. Go to [colab.research.google.com](https://colab.research.google.com) and ensure Pro+ subscription is active
2. Create a new notebook
3. Set runtime: **Runtime → Change runtime type → A100 GPU**
4. Upload the training data and fine_tuning.py script (or mount Google Drive)
5. Install dependencies in Colab:
   ```python
   !pip install -q peft transformers bitsandbytes accelerate datasets trl
   !pip install -q -e /content/earningslens  # if mounted from Drive
   ```
6. Set environment variables in Colab:
   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_..."       # Your Hugging Face token
   os.environ["OPENAI_API_KEY"] = "sk-..." # For evaluation
   ```
7. Run fine-tuning (see Section 3, Week 3 for exact commands)

> **Cost estimate:** One fine-tuning run on Colab Pro+ costs approximately $5–8 in compute units. Budget accordingly.

### 2.7 configs/config.yaml Overview

The central configuration file controls all pipeline behavior. Key sections to review before running:

```yaml
data:
  start_year: 2010        # Transcript data start year
  end_year: 2023          # Transcript data end year
  universe: "sp500"       # "sp500" or "russell1000" or "all"
  min_transcript_length: 500  # Filter very short transcripts

baseline:
  min_target_confidence: 0.7  # spaCy confidence threshold
  window_quarters: 2          # How many quarters back to compare

llm:
  model: "gpt-4o-mini"    # Override with OPENAI_MODEL env var
  temperature: 0.1         # Low temp for deterministic extraction
  max_tokens: 2048
  batch_size: 10           # Transcripts per API batch

rag:
  embedding_model: "text-embedding-3-small"
  similarity_threshold: 0.75
  top_k: 5                 # Retrieve top-5 similar past targets

evaluation:
  ff_factors: 3            # 3 or 5 factor model
  min_firm_quarters: 8     # Min observations per firm
```

Adjust `universe` to `"sp500"` for the S&P 500 subset during development (faster, cheaper), then switch to the broader universe for final results.

---

## 3. Week-by-Week Execution Plan

> **Read this section before touching the codebase.** Each day has a clear goal, exact commands, and verification steps. Do not proceed to the next day until the current day's verification passes.

### Week 1: Data Extraction & Baseline

#### Day 1–2: Environment Setup + WRDS Data Pull

**Goal:** Pull all WRDS data and have it saved in `data/raw/`.

**Day 1 Morning — Environment (all three team members):**
```bash
# Each person sets up their environment independently
git clone https://github.com/[repo]/earningslens.git
cd earningslens
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_lg
cp .env.example .env
# Edit .env with your credentials
pytest tests/test_baseline.py -v  # Must see 39 passing
```

**Day 1 Afternoon — Data Pull (Timothy runs this; others monitor):**

The DataPipeline pulls all WRDS sources sequentially. This is the longest step (~2–4 hours for full S&P 500 history):

```bash
# Run with screen or tmux so it survives terminal disconnects
screen -S wrds_pull

# Activate environment
source .venv/bin/activate

# Run the full data pipeline
# This calls transcripts, returns, fundamentals, analyst_forecasts, factors, linkers
python -m src.data_retrieval.pipeline \
  --start-year 2010 \
  --end-year 2023 \
  --universe sp500 \
  --output-dir data/raw

# Detach from screen: Ctrl+A, D
# Reattach: screen -r wrds_pull
```

**Expected output files in `data/raw/`:**

| File | Description | Expected Size |
|---|---|---|
| `transcripts_raw.parquet` | All earnings call transcripts | 2–5 GB |
| `crsp_monthly_returns.parquet` | CRSP monthly stock returns | 150–300 MB |
| `crsp_daily_returns.parquet` | CRSP daily returns (for CAR) | 500 MB–1 GB |
| `compustat_quarterly.parquet` | Compustat quarterly fundamentals | 100–200 MB |
| `ibes_forecasts.parquet` | IBES analyst earnings forecasts | 50–100 MB |
| `ff_factors.parquet` | Fama-French 3/5 factors | < 5 MB |
| `identifier_links.parquet` | CRSP-Compustat-IBES linking table | 10–20 MB |

**Verification (run after pipeline completes):**
```bash
python - <<'EOF'
import pandas as pd, os

files = {
    "transcripts_raw.parquet": (10000, None),   # (min_rows, max_rows)
    "crsp_monthly_returns.parquet": (500000, None),
    "compustat_quarterly.parquet": (100000, None),
    "ibes_forecasts.parquet": (50000, None),
    "ff_factors.parquet": (100, None),
    "identifier_links.parquet": (5000, None),
}

for fname, (min_r, max_r) in files.items():
    path = f"data/raw/{fname}"
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue
    df = pd.read_parquet(path)
    size_mb = os.path.getsize(path) / 1e6
    ok = len(df) >= min_r
    print(f"{'OK' if ok else 'WARN'}: {fname} — {len(df):,} rows, {size_mb:.1f} MB")
EOF
```

**Common WRDS Issues:**

| Problem | Cause | Fix |
|---|---|---|
| `Connection timed out` | WRDS SSH session expired | Re-run; WRDS times out after ~30 min idle. Use `screen` |
| `relation "ciq.wrds_transcript_detail" does not exist` | Missing dataset subscription | Request Capital IQ access on WRDS website |
| `SSL: CERTIFICATE_VERIFY_FAILED` | macOS SSL issue | Run `pip install certifi` and `python -m certifi` |
| Very slow query | Large date range | Split by year; see `pipeline.py` `--start-year` / `--end-year` flags |
| Empty transcripts file | Wrong universe filter | Check `configs/config.yaml` → `data.universe` setting |
| `PermissionError` on parquet write | `data/raw/` doesn't exist | `mkdir -p data/raw data/processed data/cache data/sample` |

**Day 2 — Data Validation (all three team members):**

```bash
# Run the exploration notebook to sanity-check the data
python notebooks/01_data_exploration.py

# Outputs plots and summary stats to data/processed/exploration/
# Review: data/processed/exploration/summary_stats.html
```

Key things to verify manually:
- Transcripts cover the full 2010–2023 period with no multi-year gaps
- CRSP returns have no systematic NaN stretches (check 2008, 2020 for COVID/GFC)
- Compustat fundamentals have expected columns: `gvkey`, `datadate`, `atq`, `revtq`, `epspxq`
- IBES has `ticker`, `fpedats`, `value` (analyst EPS forecast) columns

#### Day 3–4: Run spaCy Baseline Pipeline

**Goal:** Produce `spacy_targets.parquet` and `spacy_mt_scores.parquet`.

```bash
# Run the full baseline pipeline
# Processes all transcripts through SpacyTargetExtractor
# Then computes Moving Targets via MovingTargetsComputer
python -m src.baseline.baseline_pipeline \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed \
  --config configs/config.yaml \
  --log-level INFO

# This takes ~30–90 minutes depending on corpus size
# Progress is logged to logs/baseline_pipeline.log
```

**Expected output files:**

| File | Description | Key Columns |
|---|---|---|
| `data/processed/spacy_targets.parquet` | Extracted targets per transcript | `transcript_id`, `firm_id`, `quarter`, `target_text`, `target_value`, `target_type`, `confidence` |
| `data/processed/spacy_mt_scores.parquet` | MT scores per firm-quarter | `gvkey`, `quarter`, `mt_score`, `n_targets`, `direction` |

**Verification:**
```bash
python - <<'EOF'
import pandas as pd

targets = pd.read_parquet("data/processed/spacy_targets.parquet")
mt = pd.read_parquet("data/processed/spacy_mt_scores.parquet")

print(f"Targets extracted: {len(targets):,}")
print(f"Unique firms: {targets['firm_id'].nunique():,}")
print(f"MT scores computed: {len(mt):,}")
print(f"MT score distribution:\n{mt['mt_score'].describe()}")

# Check for obvious issues
assert targets['confidence'].between(0, 1).all(), "Confidence out of range"
assert not mt['mt_score'].isna().all(), "All MT scores are NaN"
print("Verification passed.")
EOF
```

**Manual Spot-Check (required — do not skip):**

Pick 5 random transcripts and read the extracted targets against the raw text:

```bash
python - <<'EOF'
import pandas as pd, random

targets = pd.read_parquet("data/processed/spacy_targets.parquet")
transcripts = pd.read_parquet("data/raw/transcripts_raw.parquet")

sample_ids = random.sample(list(targets['transcript_id'].unique()), 5)
for tid in sample_ids:
    t = targets[targets['transcript_id'] == tid]
    raw = transcripts[transcripts['transcript_id'] == tid]['text'].iloc[0]
    print(f"\n{'='*60}")
    print(f"Transcript: {tid}")
    print(f"Extracted targets ({len(t)}):")
    for _, row in t.iterrows():
        print(f"  [{row['target_type']}] {row['target_text'][:100]}")
    # Find the sentence in the raw text
    print(f"\nRaw text (first 500 chars):\n{raw[:500]}")
EOF
```

For each of the 5 transcripts, ask:
- Does the extracted target appear verbatim in the transcript?
- Is the `target_type` label correct (revenue / EPS / margin / other)?
- Are obvious targets missed? (scan the raw text for words like "expect", "guidance", "outlook", "project")
- Are obvious non-targets included? (e.g., historical statements like "last year we achieved $2B")

Document your observations in a text file at `data/processed/spot_check_notes.txt`.

#### Day 5: Manual Annotation Task

**Goal:** Produce 100 annotated transcript segments to serve as ground truth for NLP evaluation.

**Step 1: Sample 100 segments:**
```bash
python - <<'EOF'
import pandas as pd, random

random.seed(42)  # Fixed seed for reproducibility
transcripts = pd.read_parquet("data/raw/transcripts_raw.parquet")

# Sample 100 unique transcripts
sample = transcripts.sample(n=100, random_state=42)

# For each transcript, extract the Q&A section (if available)
# or the prepared remarks section
# Save as individual text segments
import os
os.makedirs("data/sample/annotation", exist_ok=True)

for i, (_, row) in enumerate(sample.iterrows()):
    # Take a 500-word window from a random position in the transcript
    words = row['text'].split()
    if len(words) > 500:
        start = random.randint(0, len(words) - 500)
        segment = " ".join(words[start:start+500])
    else:
        segment = row['text']
    
    with open(f"data/sample/annotation/segment_{i:03d}.txt", "w") as f:
        f.write(f"Transcript ID: {row['transcript_id']}\n")
        f.write(f"Firm: {row.get('firm_name', 'Unknown')}\n")
        f.write(f"Quarter: {row.get('quarter', 'Unknown')}\n")
        f.write("="*60 + "\n")
        f.write(segment)

print("Saved 100 annotation segments to data/sample/annotation/")
EOF
```

**Step 2: Divide segments among team members:**
- Timothy (tc3460): segments 000–032 (33 segments)
- Yewen (yl5888): segments 033–065 (33 segments)
- Tiantian (th3166): segments 066–099 (34 segments)

**Step 3: Annotation guidelines.**

For each segment, create a file `segment_NNN_annotated.txt` with the following format:

```
Transcript ID: [from header]
Annotator: [your UNI]
Date: [today's date]

TARGETS FOUND:
---
Target 1:
  Text: [exact quote from transcript]
  Type: [REVENUE | EPS | MARGIN | VOLUME | GROWTH | OTHER]
  Value: [numeric value if present, else "qualitative"]
  Direction: [UP | DOWN | STABLE | UNCLEAR]
  Time Horizon: [NEXT_QUARTER | NEXT_YEAR | MULTI_YEAR | UNSPECIFIED]
  Confidence: [HIGH | MEDIUM | LOW]

Target 2:
  [repeat structure]
  
NO_TARGETS: [YES if no targets found in this segment]
NOTES: [any observations about ambiguous cases]
```

**What counts as a performance target:**
- Explicit forward-looking numeric guidance ("we expect EPS of $1.20")
- Range guidance ("revenue between $2.0B and $2.2B")
- Qualitative directional guidance ("we anticipate margin expansion next year")
- Conditional guidance ("if demand holds, we should hit $500M")

**What does NOT count:**
- Historical statements ("last quarter we achieved $1.8B")
- Industry-level commentary ("the sector is growing at 5%")
- Purely backward-looking comparisons without a forward component

**Step 4: Inter-annotator agreement.**

Each person annotates 5 segments from another person's set (for overlap). Use these overlapping annotations to compute Cohen's Kappa:

```bash
python - <<'EOF'
# After annotation is complete
from sklearn.metrics import cohen_kappa_score

# Load annotations from overlap segments
# Fill in your actual annotation data here
annotator_a = [1, 1, 0, 1, 0]  # 1=target present, 0=no target
annotator_b = [1, 0, 0, 1, 0]

kappa = cohen_kappa_score(annotator_a, annotator_b)
print(f"Cohen's Kappa: {kappa:.3f}")
# > 0.6 is acceptable, > 0.8 is strong agreement
EOF
```

Report Kappa in the final paper. If Kappa < 0.6, resolve disagreements through discussion and re-annotate.

---

### Week 2: LLM Extraction & RAG

#### Day 1–2: Run LLM Extraction Pipeline

**Goal:** Produce `llm_targets.parquet` with richer, LLM-extracted targets.

**Important: Cost management.** Before running on the full corpus, test on a small subset:

```bash
# Test on 10 transcripts first
python -m src.llm_extraction.extraction_pipeline \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed \
  --limit 10 \
  --config configs/config.yaml

# Review the output
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/processed/llm_targets.parquet")
print(df.head(20).to_string())
print(f"\nColumns: {list(df.columns)}")
EOF
```

If the 10-transcript test looks correct, run the full pipeline:

```bash
# Full pipeline on S&P 500 subset (recommended starting point)
# Expected cost: $10–15 for S&P 500 history with gpt-4o-mini
python -m src.llm_extraction.extraction_pipeline \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed \
  --config configs/config.yaml \
  --checkpoint-every 100 \
  --log-level INFO
```

The `--checkpoint-every 100` flag saves progress every 100 transcripts. If the pipeline crashes, it resumes from the last checkpoint automatically — this is important for API resiliency.

**Monitor API costs in real time:**
```bash
# In a separate terminal, watch the cost log
tail -f logs/llm_extraction.log | grep "cost"
```

**Expected output:**

| File | Description |
|---|---|
| `data/processed/llm_targets.parquet` | LLM-extracted targets (richer than spaCy) |
| `data/processed/llm_extraction_costs.json` | Per-batch API cost tracking |
| `data/processed/llm_mt_scores.parquet` | LLM-based MT scores |

**Comparing LLM vs. spaCy outputs:**
```bash
python - <<'EOF'
import pandas as pd

spacy_t = pd.read_parquet("data/processed/spacy_targets.parquet")
llm_t   = pd.read_parquet("data/processed/llm_targets.parquet")

# Coverage comparison
spacy_firms = set(spacy_t['firm_id'].unique())
llm_firms   = set(llm_t['firm_id'].unique())
print(f"spaCy covered firms: {len(spacy_firms)}")
print(f"LLM covered firms:   {len(llm_firms)}")
print(f"In LLM but not spaCy: {len(llm_firms - spacy_firms)}")

# Average targets per transcript
print(f"\nAvg targets/transcript — spaCy: {spacy_t.groupby('transcript_id').size().mean():.2f}")
print(f"Avg targets/transcript — LLM:   {llm_t.groupby('transcript_id').size().mean():.2f}")

# Type distribution
print(f"\nspaCy target types:\n{spacy_t['target_type'].value_counts()}")
print(f"\nLLM target types:\n{llm_t['target_type'].value_counts()}")
EOF
```

#### Day 3–4: ChromaDB Vector Index & Semantic Matching

**Goal:** Build the ChromaDB index and compute LLM-based MT scores using semantic similarity.

**Step 1: Build the vector index:**
```bash
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from src.rag.vector_store import TargetVectorStore
from src.utils.config_loader import load_config
import pandas as pd

cfg = load_config()
store = TargetVectorStore(persist_dir=cfg["rag"]["chroma_persist_dir"])

# Load all LLM-extracted targets
targets = pd.read_parquet("data/processed/llm_targets.parquet")
print(f"Indexing {len(targets):,} targets...")

# Build the full index (this may take 10–30 minutes)
store.build_full_index(targets)
print(f"Index built. Collection size: {store.collection.count()}")
EOF
```

**Step 2: Calibrate similarity threshold on annotation set.**

The `similarity_threshold` in `configs/config.yaml` (default 0.75) determines when two targets across quarters are considered "the same target." Use the annotated segments to calibrate this:

```bash
python - <<'EOF'
from src.rag.semantic_matcher import SemanticContinuityMatcher
from src.utils.config_loader import load_config
import pandas as pd, numpy as np

cfg = load_config()
matcher = SemanticContinuityMatcher(config=cfg)

# Load annotation data (must have ground-truth cross-quarter match labels)
# You should have created these during Week 1 annotation
ann = pd.read_csv("data/sample/annotation/cross_quarter_matches.csv")

# Test thresholds from 0.6 to 0.9
thresholds = np.arange(0.60, 0.91, 0.05)
results = []
for t in thresholds:
    cfg["rag"]["similarity_threshold"] = float(t)
    # Compute precision and recall on annotation set
    # (implementation depends on annotation format)
    precision, recall, f1 = matcher.evaluate_on_annotations(ann, threshold=t)
    results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    print(f"Threshold {t:.2f}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

# Pick threshold with best F1
best = max(results, key=lambda x: x["f1"])
print(f"\nBest threshold: {best['threshold']:.2f} (F1={best['f1']:.3f})")
EOF
```

Update `configs/config.yaml` with the best threshold before running semantic MT computation.

**Step 3: Run semantic MT computation:**
```bash
python - <<'EOF'
from src.rag.semantic_matcher import SemanticContinuityMatcher
from src.utils.config_loader import load_config
import pandas as pd

cfg = load_config()
matcher = SemanticContinuityMatcher(config=cfg)

targets = pd.read_parquet("data/processed/llm_targets.parquet")
mt_scores = matcher.compute_mt_scores(targets)
mt_scores.to_parquet("data/processed/llm_mt_scores.parquet", index=False)
print(f"Computed {len(mt_scores):,} LLM MT scores")
print(mt_scores.describe())
EOF
```

#### Day 5: LangGraph Pipeline End-to-End

**Goal:** Run the full 4-agent LangGraph pipeline on a representative subset to verify agent interactions.

```bash
# Run on a 50-firm subset for speed
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from src.agents.graph import build_pipeline_graph
from src.utils.config_loader import load_config
import pandas as pd

cfg = load_config()
graph = build_pipeline_graph(config=cfg)

# Load a subset of transcripts (50 firms, all quarters)
transcripts = pd.read_parquet("data/raw/transcripts_raw.parquet")
firms_subset = transcripts['firm_id'].unique()[:50]
subset = transcripts[transcripts['firm_id'].isin(firms_subset)]

print(f"Running LangGraph pipeline on {len(subset)} transcripts from {len(firms_subset)} firms...")

results = []
for _, row in subset.iterrows():
    state = graph.invoke({
        "transcript_id": row["transcript_id"],
        "text": row["text"],
        "firm_id": row["firm_id"],
        "quarter": row["quarter"],
    })
    results.append(state)

import json
with open("data/processed/langgraph_subset_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Pipeline complete. Results saved.")
# Review the first result
print(json.dumps(results[0], indent=2, default=str)[:1000])
EOF
```

**Verify each agent produced output:**
```bash
python - <<'EOF'
import json

with open("data/processed/langgraph_subset_results.json") as f:
    results = json.load(f)

r = results[0]  # Check first result
assert "extracted_targets" in r,  "Extractor agent failed"
assert "matched_targets" in r,     "Comparator agent failed"
assert "classified_revision" in r, "Classifier agent failed"
assert "mt_score" in r,            "Reporter agent failed"

print("All 4 agents produced output for at least the first transcript.")
print(f"Extractor found {len(r['extracted_targets'])} targets")
print(f"Comparator matched {len(r['matched_targets'])} cross-quarter pairs")
print(f"Classifier: {r['classified_revision']}")
print(f"Reporter MT score: {r['mt_score']:.4f}")
EOF
```

---

### Week 3: Evaluation & Fine-Tuning

#### Day 1–2: Fama-MacBeth Regressions

**Goal:** Run the main empirical test — does MT predict next-month returns?

**Step 1: Merge all data into a panel:**
```bash
python - <<'EOF'
from src.data_retrieval.pipeline import DataPipeline
from src.utils.config_loader import load_config
import pandas as pd

cfg = load_config()
pipeline = DataPipeline(config=cfg)

# merge_all() joins MT scores with returns, fundamentals, factors, IBES
panel_spacy = pipeline.merge_all(
    mt_scores_path="data/processed/spacy_mt_scores.parquet"
)
panel_llm = pipeline.merge_all(
    mt_scores_path="data/processed/llm_mt_scores.parquet"
)

panel_spacy.to_parquet("data/processed/panel_spacy.parquet", index=False)
panel_llm.to_parquet("data/processed/panel_llm.parquet", index=False)

print(f"spaCy panel: {panel_spacy.shape}")
print(f"LLM panel:   {panel_llm.shape}")
print(f"Columns: {list(panel_spacy.columns)}")
EOF
```

**Step 2: Run Fama-MacBeth regressions:**
```bash
python - <<'EOF'
from src.evaluation.fama_macbeth import FamaMacBeth
import pandas as pd

fm = FamaMacBeth()

for label, path in [("spaCy", "data/processed/panel_spacy.parquet"),
                     ("LLM",   "data/processed/panel_llm.parquet")]:
    panel = pd.read_parquet(path)
    
    # Run the full FM regression
    # Controls: log(Size), BM, Momentum, SUE
    results = fm.run(
        panel=panel,
        dependent_var="ret_next_month",
        test_var="mt_score",
        controls=["log_size", "bm", "mom12", "sue"],
        ff_adjust=True,   # Use FF3 excess returns
        newey_west_lags=3
    )
    
    results.to_csv(f"data/processed/fm_results_{label.lower()}.csv")
    print(f"\n{'='*50}")
    print(f"Fama-MacBeth Results — {label}")
    print(f"{'='*50}")
    print(results.to_string())
    
    # Key metric: MT coefficient and its t-stat
    mt_row = results[results['variable'] == 'mt_score'].iloc[0]
    print(f"\nMT coefficient: {mt_row['coef']:.4f}")
    print(f"Newey-West t-stat: {mt_row['t_stat']:.2f}")
    print(f"Annualized alpha: {mt_row['coef'] * 12 * 100:.2f}%")
EOF
```

**Interpreting the output** — See Section 6 for detailed interpretation guidance. Briefly:
- MT coefficient > 0 and t-stat > 2.0 replicates the paper's main finding
- Compare spaCy t-stat vs LLM t-stat — a higher LLM t-stat is the paper's core contribution

**Step 3: Quintile portfolio sorts:**
```bash
python - <<'EOF'
from src.evaluation.portfolio_construction import QuintilePortfolio
import pandas as pd

for label, path in [("spaCy", "data/processed/panel_spacy.parquet"),
                     ("LLM",   "data/processed/panel_llm.parquet")]:
    panel = pd.read_parquet(path)
    qp = QuintilePortfolio()
    alpha_results = qp.compute_calendar_time_alpha(
        panel=panel,
        signal_col="mt_score",
        return_col="ret_next_month",
        factors_path="data/raw/ff_factors.parquet",
        n_quintiles=5
    )
    alpha_results.to_csv(f"data/processed/portfolio_alpha_{label.lower()}.csv")
    print(f"\n{label} — Q5-Q1 Long-Short Alpha:")
    print(alpha_results[["alpha_monthly", "t_stat", "sharpe"]].to_string())
EOF
```

**Step 4: CAR analysis around announcement dates:**
```bash
python -m src.evaluation.announcement_cars \
  --panel data/processed/panel_llm.parquet \
  --daily-returns data/raw/crsp_daily_returns.parquet \
  --output data/processed/car_results.parquet \
  --window-pre 1 \
  --window-post 3
```

#### Day 3–4: Ablation Studies

**Goal:** Decompose which aspects of the LLM pipeline drive performance improvement.

**Ablation 1: Model comparison (GPT-4o vs GPT-4o-mini vs spaCy):**
```bash
# Run extraction with GPT-4o (more expensive — use 100-firm subset)
python -m src.llm_extraction.extraction_pipeline \
  --backend openai \
  --model gpt-4o \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed/ablation \
  --limit-firms 100 \
  --suffix _gpt4o

# Compare performance metrics
python -m src.evaluation.comparison \
  --annotation-dir data/sample/annotation \
  --predictions data/processed/ablation/llm_targets_gpt4o.parquet \
                data/processed/spacy_targets.parquet \
  --labels "GPT-4o" "spaCy" \
  --output data/processed/ablation/model_comparison.csv
```

**Ablation 2: Prompting strategy comparison:**
The `src/llm_extraction/prompts.py` contains multiple prompt variants. Edit `configs/config.yaml` to switch between them:

```yaml
llm:
  prompt_strategy: "chain_of_thought"   # Options: "zero_shot", "few_shot", "chain_of_thought"
```

Run extraction with each strategy on the same 100-firm subset, then compare F1 scores against the annotation set.

**Ablation 3: RAG retrieval strategy:**
```yaml
rag:
  retrieval_strategy: "semantic"   # Options: "exact", "semantic", "hybrid"
  top_k: 5                         # Test: 1, 3, 5, 10
```

For each combination, compute FM regression t-stats to see which retrieval strategy produces the strongest return-predictive signal.

**Compile ablation table:**
```bash
python - <<'EOF'
import pandas as pd

# Load all ablation results
ablation_configs = [
    ("spaCy Baseline", "data/processed/fm_results_spacy.csv"),
    ("GPT-4o-mini (zero-shot)", "data/processed/ablation/fm_zero_shot.csv"),
    ("GPT-4o-mini (few-shot)", "data/processed/ablation/fm_few_shot.csv"),
    ("GPT-4o-mini (CoT)", "data/processed/fm_results_llm.csv"),
    ("GPT-4o (CoT)", "data/processed/ablation/fm_gpt4o.csv"),
]

rows = []
for label, path in ablation_configs:
    try:
        df = pd.read_csv(path)
        mt_row = df[df['variable'] == 'mt_score'].iloc[0]
        rows.append({
            "Model": label,
            "MT Coef": f"{mt_row['coef']:.4f}",
            "t-stat": f"{mt_row['t_stat']:.2f}",
            "R²": f"{mt_row.get('r2', float('nan')):.3f}",
        })
    except Exception as e:
        rows.append({"Model": label, "Error": str(e)})

ablation_df = pd.DataFrame(rows)
print(ablation_df.to_string(index=False))
ablation_df.to_csv("data/processed/ablation_table.csv", index=False)
EOF
```

#### Day 5: QLoRA Fine-Tuning (Stretch Goal)

This is a stretch goal — only pursue if the main evaluation is complete.

**Upload to Colab:**
1. Zip your annotation data and fine_tuning.py:
   ```bash
   zip -r colab_finetune.zip data/sample/annotation/ src/llm_extraction/fine_tuning.py
   ```
2. Upload to Google Drive, then mount in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

**Run fine-tuning in Colab:**
```python
import os
os.chdir("/content/drive/MyDrive/earningslens")

# Install dependencies
!pip install -q peft transformers bitsandbytes accelerate datasets trl

# Run fine-tuning
!python src/llm_extraction/fine_tuning.py \
  --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
  --training-data data/sample/annotation \
  --output-dir /content/drive/MyDrive/mistral-earningslens \
  --num-epochs 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --gradient-accumulation-steps 4
```

**Expected training time:** 45–90 minutes on A100.

**Evaluate fine-tuned model:**
```bash
python -m src.llm_extraction.extraction_pipeline \
  --backend local \
  --model-path /path/to/mistral-earningslens \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed/ablation \
  --limit-firms 50 \
  --suffix _mistral_finetuned
```

---

### Week 4: Demo & Report

#### Day 1–2: Gradio Demo Setup

**Goal:** Get `demo/app.py` running robustly for a live class demonstration.

**Pre-compute results for demo robustness** (critical — live API calls during demos are risky):
```bash
# Pre-compute LLM extraction for a set of "demo firms"
# Pick 5–10 well-known companies for the demo
DEMO_FIRMS="AAPL,MSFT,AMZN,GOOGL,META,TSLA,JPM,JNJ,XOM,WMT"

python - <<'EOF'
import pandas as pd

transcripts = pd.read_parquet("data/raw/transcripts_raw.parquet")
demo_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "XOM", "WMT"]

demo_data = transcripts[transcripts['ticker'].isin(demo_tickers)]
demo_data.to_parquet("data/cache/demo_transcripts.parquet", index=False)
print(f"Saved {len(demo_data)} transcripts for {len(demo_tickers)} demo firms")
EOF

# Pre-run extraction on demo transcripts
python -m src.llm_extraction.extraction_pipeline \
  --backend openai \
  --model gpt-4o-mini \
  --input data/cache/demo_transcripts.parquet \
  --output-dir data/cache \
  --suffix _demo

# Pre-build ChromaDB index for demo firms only
python - <<'EOF'
from src.rag.vector_store import TargetVectorStore
import pandas as pd

store = TargetVectorStore(persist_dir="data/cache/chromadb_demo")
demo_targets = pd.read_parquet("data/cache/llm_targets_demo.parquet")
store.build_full_index(demo_targets)
print(f"Demo index built: {store.collection.count()} targets")
EOF
```

**Launch the demo:**
```bash
python demo/app.py \
  --demo-mode precomputed \
  --targets-path data/cache/llm_targets_demo.parquet \
  --mt-scores-path data/processed/llm_mt_scores.parquet \
  --chroma-dir data/cache/chromadb_demo \
  --port 7860 \
  --share   # Generates a public URL for remote viewing
```

**Test both demo modes:**
- **Mode 1 (Live):** Paste a raw transcript excerpt → watch the 4-agent pipeline extract targets in real time
- **Mode 2 (Precomputed):** Select a ticker and date range → see MT scores, portfolio performance, and regression coefficients

**Record demo walkthrough:**
Use QuickTime (macOS) or OBS (Windows/Linux) to record a 3–5 minute walkthrough covering both modes. Save to `demo/earningslens_demo.mp4`.

#### Day 3–5: Final Report Writing

**Report structure** (aligned with typical NLP/finance paper format):

```
1. Introduction (1 page)
   - Research question, motivation, contribution
   - Summary of findings: does LLM improve return predictability?

2. Related Work (0.5 page)
   - Cohen & Nguyen (2024) — the paper being replicated
   - Earnings call NLP literature
   - LLM for financial NLP

3. Data (0.5 page)
   - WRDS Capital IQ transcripts
   - CRSP/Compustat/IBES
   - Summary statistics table (Table 1)

4. Methodology (2 pages)
   4.1 spaCy Baseline (Layer 1)
   4.2 LLM Extraction with CoT (Layer 2)
   4.3 RAG-based Cross-Quarter Matching
   4.4 LangGraph Pipeline
   4.5 Fama-MacBeth Regression Setup

5. Results (2 pages)
   5.1 NLP Evaluation (Table 2: Precision/Recall/F1)
   5.2 Fama-MacBeth Main Results (Table 3)
   5.3 Portfolio Alpha (Table 4)
   5.4 Ablation Studies (Table 5)
   5.5 CAR Analysis (Figure 1)

6. Discussion (0.5 page)
   - Why LLM improves (or doesn't improve) over spaCy
   - Limitations: sample period, API costs, prompt sensitivity

7. Conclusion (0.5 page)

8. References
```

**Generate LaTeX tables from code:**
```bash
python - <<'EOF'
from src.evaluation.comparison import SignalComparison
import pandas as pd

comp = SignalComparison()

# Table 2: NLP metrics
nlp_table = comp.nlp_metrics_table(
    annotation_dir="data/sample/annotation",
    spacy_predictions="data/processed/spacy_targets.parquet",
    llm_predictions="data/processed/llm_targets.parquet"
)
nlp_table.to_latex("docs/table_nlp_metrics.tex", index=False, float_format="%.3f")
print("Table 2 (NLP metrics) saved to docs/table_nlp_metrics.tex")

# Table 3: Fama-MacBeth
for label in ["spacy", "llm"]:
    df = pd.read_csv(f"data/processed/fm_results_{label}.csv")
    df.to_latex(f"docs/table_fm_{label}.tex", index=False, float_format="%.4f")
print("Table 3 (FM results) saved")

# Table 5: Ablation
abl = pd.read_csv("data/processed/ablation_table.csv")
abl.to_latex("docs/table_ablation.tex", index=False)
print("Table 5 (Ablation) saved")
EOF
```

---

### Week 5: Polish & Present

#### Day 1–2: GitHub Repository Polish

**Checklist for clean repository:**

```bash
# Verify .gitignore excludes sensitive/large files
cat .gitignore | grep -E "\.env|data/raw|__pycache__|\.venv"

# Check that no API keys are committed
git log --all --full-history -- .env
git grep -r "sk-" -- "*.py" "*.yaml"  # Should return nothing

# Add commit history cleaning (if needed)
# If API keys were ever committed, use git-filter-repo to remove them

# Final commit checklist
git status   # No untracked sensitive files
git log --oneline -10  # Commit messages should be meaningful

# Tag the final submission
git tag -a "final-submission" -m "STAT GR5293 Final Submission"
```

**README.md must contain:**
- Project title and one-paragraph description
- Architecture diagram (can copy the ASCII art from this guidebook)
- Installation instructions (condensed from Section 2 of this guidebook)
- Quick-start example showing the end-to-end pipeline in 5 commands
- Links to the paper, WRDS, and key dependencies
- Team member names and UNIs
- Course and semester

**Code quality:**
```bash
# Format code with black
black src/ demo/ --line-length 100

# Lint with flake8
flake8 src/ demo/ --max-line-length 100 --ignore E203,W503

# All tests still pass
pytest tests/test_baseline.py -v
```

#### Day 3: Presentation Slides

**Suggested slide structure (12–15 slides, 10 minutes):**

```
Slide 1:  Title, team, course
Slide 2:  Motivation — why earnings call targets matter
Slide 3:  Cohen & Nguyen (2024) — what we replicate
Slide 4:  EarningsLens system architecture (3 layers diagram)
Slide 5:  Layer 1 — spaCy baseline: how it works + example output
Slide 6:  Layer 2 — LLM extraction: CoT prompt example, example output
Slide 7:  Layer 2 — RAG + LangGraph pipeline diagram
Slide 8:  Data: transcript example + summary statistics (Table 1)
Slide 9:  Results — NLP metrics table (Table 2): LLM beats spaCy on F1
Slide 10: Results — Fama-MacBeth (Table 3): LLM t-stat > spaCy t-stat
Slide 11: Results — Portfolio alpha (Table 4): L-S alpha chart
Slide 12: Results — Ablation (Table 5): CoT > few-shot > zero-shot > spaCy
Slide 13: Live Demo (transition to Gradio app)
Slide 14: Limitations and future work
Slide 15: Conclusions
```

**Key talking points to prepare:**
- Be ready to explain what "Fama-MacBeth regression" is to non-finance classmates (it's just a two-step procedure that estimates coefficients cross-sectionally each month, then averages them)
- Be ready to explain why RAG is better than simple string matching for cross-quarter comparison
- Be ready to discuss API cost trade-offs: GPT-4o vs GPT-4o-mini
- Have the demo pre-launched and the browser tab open before presentation

#### Day 4–5: Practice and Refinement

- Do at least one full dry run with the actual demo environment
- Time the presentation — 10 minutes talk + 5 minutes demo is typical
- Assign speaking roles: who introduces the problem, who explains the system, who runs the demo, who covers results
- Prepare for questions: Why not fine-tune on more data? How would this generalize to non-S&P500 firms? What's the economic explanation for the MT signal?

---

## 4. Module-by-Module User Guide

### 4.1 `src/data_retrieval/transcripts.py`

**What it does:** Connects to WRDS Capital IQ and pulls earnings call transcript text for a specified set of firms and date range.

**How to run:**
```bash
python - <<'EOF'
from src.data_retrieval.transcripts import TranscriptRetriever
from src.utils.config_loader import load_config

cfg = load_config()
retriever = TranscriptRetriever(config=cfg)

# Pull all S&P 500 transcripts 2010–2023
df = retriever.fetch_all(
    universe="sp500",
    start_year=2010,
    end_year=2023
)
df.to_parquet("data/raw/transcripts_raw.parquet", index=False)
print(f"Fetched {len(df):,} transcripts")
EOF
```

**Input:** WRDS connection, config parameters  
**Output:** `transcripts_raw.parquet` with columns: `transcript_id`, `gvkey`, `ticker`, `firm_name`, `quarter`, `year`, `call_date`, `text`, `section` (prepared_remarks / qa)

**Key parameters:**
- `universe`: `"sp500"`, `"russell1000"`, or `"all"`
- `start_year`, `end_year`: Date range
- `min_length`: Filter transcripts shorter than N characters (default 500)

**Verify:** `df['text'].str.len().describe()` — median length should be ~10,000–50,000 characters

**Common errors:**
- `AttributeError: 'NoneType' object has no attribute 'raw_sql'` — WRDS connection failed; re-run `wrds.Connection()`
- `KeyError: 'ciq'` — You don't have Capital IQ access; request it on WRDS

### 4.2 `src/data_retrieval/returns.py`

**What it does:** Pulls CRSP monthly and daily stock returns, including market cap data for size controls.

**How to run:**
```bash
python - <<'EOF'
from src.data_retrieval.returns import ReturnRetriever
from src.utils.config_loader import load_config

cfg = load_config()
ret = ReturnRetriever(config=cfg)
monthly = ret.fetch_monthly(start_year=2010, end_year=2024)
daily = ret.fetch_daily(start_year=2010, end_year=2024)
monthly.to_parquet("data/raw/crsp_monthly_returns.parquet", index=False)
daily.to_parquet("data/raw/crsp_daily_returns.parquet", index=False)
EOF
```

**Output columns (monthly):** `permno`, `date`, `ret`, `mktcap`, `shrout`, `prc`  
**Output columns (daily):** `permno`, `date`, `ret`, `vol`

**Key note:** Returns are raw CRSP returns (including dividends). The pipeline converts to excess returns by subtracting the risk-free rate from FF factors.

### 4.3 `src/data_retrieval/fundamentals.py`

**What it does:** Pulls Compustat quarterly financial statement data for computing Size, Book-to-Market, and other control variables.

**Output columns:** `gvkey`, `datadate`, `atq` (total assets), `ceqq` (book equity), `revtq` (revenue), `epspxq` (EPS), `cshoq` (shares outstanding), `prccq` (price)

**Derived variables computed in `pipeline.py`:**
- `bm` = `ceqq / (prccq * cshoq)` — Book-to-Market ratio
- `log_size` = `log(prccq * cshoq)` — Log market capitalization
- `roe` = `epspxq / (ceqq / 4)` — Return on Equity

### 4.4 `src/data_retrieval/analyst_forecasts.py`

**What it does:** Pulls IBES consensus analyst EPS forecasts and computes Standardized Unexpected Earnings (SUE).

**SUE formula:**
```
SUE = (Actual_EPS - Consensus_EPS) / std(Forecast_Errors over past 8 quarters)
```

**Output:** `ibes_forecasts.parquet` with `ticker`, `fpedats`, `value`, `actual`, `sue`

### 4.5 `src/data_retrieval/pipeline.py` — DataPipeline

**What it does:** Orchestrates all retrieval modules and merges data into a unified panel.

**Key method — `merge_all()`:**
```python
# Joins on: gvkey (Compustat), permno (CRSP), ticker (IBES)
# Aligns time: MT scores from quarter Q are matched to returns in Q+1
# to avoid look-ahead bias
panel = pipeline.merge_all(mt_scores_path="data/processed/spacy_mt_scores.parquet")
```

**Look-ahead bias protection:** The merge explicitly offsets MT scores by 1 quarter relative to returns. Always check that `ret_next_month` in the panel corresponds to returns after the earnings call date.

### 4.6 `src/baseline/target_extractor.py` — SpacyTargetExtractor

**What it does:** Applies spaCy's dependency parser and NER to identify performance target sentences in transcript text.

**How to run:**
```python
from src.baseline.target_extractor import SpacyTargetExtractor

extractor = SpacyTargetExtractor(model="en_core_web_lg", confidence_threshold=0.7)
targets = extractor.extract("We expect revenue to reach $2.5 billion next quarter.")
# Returns: [{"text": "We expect revenue to reach $2.5 billion next quarter.",
#            "target_type": "REVENUE", "value": 2500000000, "confidence": 0.89}]
```

**Key parameters (in config.yaml):**
- `baseline.min_target_confidence`: Minimum spaCy confidence (0.0–1.0). Higher = fewer but more precise extractions.
- `baseline.window_quarters`: How many previous quarters to compare against for MT computation.

**What to check:** Run on 10 hand-selected sentences (5 that are clearly targets, 5 that are not). Verify the extractor catches the real targets and rejects the non-targets.

**Common errors:**
- `OSError: [E050] Can't find model 'en_core_web_lg'` — Run `python -m spacy download en_core_web_lg`

### 4.7 `src/baseline/moving_targets.py` — MovingTargetsComputer

**What it does:** Computes the MT score for each firm-quarter by comparing current targets to prior-quarter targets.

**MT Score formula:**
```
For each target type T in quarter Q:
  delta_T = (value_T_Q - value_T_{Q-1}) / |value_T_{Q-1}|   # Normalized change

MT_Q = weighted_average(delta_T) over all target types
     where weights = (confidence_T_Q * coverage_T_Q)

Coverage_T_Q = 1 if type T was mentioned both quarters, else 0
```

**The MT score is positive** when targets increase quarter-over-quarter, and negative when they decrease.

### 4.8 `src/llm_extraction/prompts.py`

**What it does:** Contains all prompt templates for LLM-based extraction. Three strategies:

1. **Zero-shot:** Direct instruction with no examples
2. **Few-shot:** 3 annotated examples inline in the prompt
3. **Chain-of-thought (default):** Asks the LLM to reason step by step before outputting structured JSON

**Viewing and modifying prompts:**
```bash
python - <<'EOF'
from src.llm_extraction.prompts import EXTRACTION_PROMPTS
for name, template in EXTRACTION_PROMPTS.items():
    print(f"\n{'='*40}")
    print(f"Strategy: {name}")
    print(template[:300])
EOF
```

**When to modify:** If NLP evaluation shows low recall, try relaxing the CoT reasoning step by removing constraints. If precision is low, add stricter output format instructions.

### 4.9 `src/llm_extraction/llm_extractor.py` — LLMTargetExtractor

**What it does:** Sends transcript text to OpenAI API (asynchronously, in batches) and parses structured JSON output.

**Key design decisions:**
- **Async:** Uses `asyncio` for concurrent API calls within a batch — respects rate limits via `asyncio.Semaphore`
- **Structured output:** Forces JSON output format using OpenAI's `response_format={"type": "json_object"}` parameter
- **Retry logic:** Exponential backoff on rate limit errors (HTTP 429)

**Running directly:**
```python
import asyncio
from src.llm_extraction.llm_extractor import LLMTargetExtractor

extractor = LLMTargetExtractor(model="gpt-4o-mini", prompt_strategy="chain_of_thought")
result = asyncio.run(extractor.extract_single(
    text="We expect full-year revenue of approximately $10 billion, representing 8% growth.",
    transcript_id="test_001"
))
print(result)
```

### 4.10 `src/rag/vector_store.py` — TargetVectorStore

**What it does:** Manages a ChromaDB collection of target embeddings. Targets are embedded using OpenAI `text-embedding-3-small` and stored with metadata (firm, quarter, type).

**Key methods:**
```python
store = TargetVectorStore(persist_dir="data/cache/chromadb")

# Build index from scratch
store.build_full_index(targets_df)  # ~10-30 minutes

# Query: find similar past targets for a given firm
similar = store.query_firm_history(
    firm_id="AAPL",
    target_text="We expect iPhone revenue of $50 billion next quarter",
    target_type="REVENUE",
    n_results=5,
    exclude_current_quarter="2023Q4"
)
```

**Persistence:** ChromaDB automatically persists to `CHROMA_PERSIST_DIR`. You don't need to rebuild the index each run — check if it already exists:
```python
if store.collection.count() > 0:
    print(f"Index already has {store.collection.count()} documents — skipping rebuild")
```

### 4.11 `src/agents/graph.py` — LangGraph Pipeline

**What it does:** Assembles the 4-agent LangGraph StateGraph and provides an `invoke()` method.

**Agent flow:**
```
Transcript text
    ↓
[Extractor Agent]        → Calls LLMTargetExtractor, stores in PipelineState.extracted_targets
    ↓
[Comparator Agent]       → Queries ChromaDB for cross-quarter matches, stores in .matched_targets
    ↓
[Classifier Agent]       → Classifies each match as UP/DOWN/STABLE/NEW/DISCONTINUED
    ↓
[Reporter Agent]         → Aggregates into final MT score, generates natural-language summary
    ↓
Final PipelineState (mt_score, summary, all intermediate outputs)
```

**Adding a new agent (e.g., a Sentiment Agent):**
1. Create `src/agents/sentiment_agent.py` with a `sentiment_node(state: PipelineState) -> PipelineState` function
2. Add the new field to `PipelineState` in `src/agents/state.py`
3. Add the node to the graph in `src/agents/graph.py`: `graph.add_node("sentiment", sentiment_node)`
4. Add the edge: `graph.add_edge("reporter", "sentiment")`

### 4.12 `src/evaluation/fama_macbeth.py` — FamaMacBeth

**What it does:** Runs Fama-MacBeth two-step cross-sectional regressions with Newey-West standard errors.

**Statistical procedure:**
1. **Each month t:** Run cross-sectional OLS of `ret_{i,t+1}` on `MT_{i,t}` and controls
2. **Time series:** Collect monthly coefficient estimates `{β_t}`
3. **Final estimate:** `β̄ = mean(β_t)`, `SE = std(β_t)/sqrt(T)` (Newey-West adjusted)
4. **t-stat:** `t = β̄ / SE`

**Key parameters:**
- `newey_west_lags`: Number of lags for Newey-West correction (default 3 for quarterly earnings data)
- `ff_adjust`: Whether to use FF3 excess returns as dependent variable
- `min_firm_quarters`: Drop firms with fewer than N observations (default 8)

### 4.13 `demo/app.py` — Gradio Demo

**What it does:** Two-panel Gradio interface.

**Tab 1 — Live Extraction:**
- User pastes transcript text
- Runs LangGraph pipeline (uses precomputed ChromaDB for cross-quarter comparison)
- Shows step-by-step agent outputs
- Displays MT score and interpretation

**Tab 2 — Historical Analysis:**
- User selects ticker and quarter range from dropdown
- Shows MT score time series chart
- Shows Fama-MacBeth regression snippet
- Shows portfolio quintile performance

**Launching for development:**
```bash
python demo/app.py --dev-mode --port 7860
```

**Launching for class demo:**
```bash
python demo/app.py \
  --demo-mode precomputed \
  --targets-path data/cache/llm_targets_demo.parquet \
  --port 7860 \
  --share
```

---

## 5. Data Dictionary

### 5.1 Raw Data Tables (WRDS Sources)

#### `transcripts_raw.parquet`

| Column | Type | Source | Description |
|---|---|---|---|
| `transcript_id` | str | CIQ | Unique transcript identifier |
| `gvkey` | str | CIQ/Compustat | Compustat firm key |
| `ticker` | str | CIQ | Stock ticker at call date |
| `firm_name` | str | CIQ | Company name |
| `call_date` | date | CIQ | Date of earnings call |
| `quarter` | str | Derived | Quarter label (e.g., "2023Q4") |
| `year` | int | Derived | Fiscal year |
| `fiscal_quarter` | int | CIQ | Fiscal quarter (1–4) |
| `text` | str | CIQ | Full transcript text |
| `section` | str | CIQ | "prepared_remarks" or "qa" |
| `word_count` | int | Derived | Number of words in text |

#### `crsp_monthly_returns.parquet`

| Column | Type | Source | Description |
|---|---|---|---|
| `permno` | int | CRSP | CRSP permanent firm identifier |
| `date` | date | CRSP | Month-end date |
| `ret` | float | CRSP | Monthly raw return (decimal) |
| `mktcap` | float | CRSP | Market cap = `abs(prc) * shrout` (thousands) |
| `prc` | float | CRSP | Closing price (negative = average of bid/ask) |
| `shrout` | float | CRSP | Shares outstanding (thousands) |
| `exchcd` | int | CRSP | Exchange code (1=NYSE, 2=AMEX, 3=NASDAQ) |

#### `compustat_quarterly.parquet`

| Column | Type | Source | Description |
|---|---|---|---|
| `gvkey` | str | Compustat | Firm identifier |
| `datadate` | date | Compustat | Quarter-end date |
| `atq` | float | Compustat | Total assets ($M) |
| `ceqq` | float | Compustat | Book equity ($M) |
| `revtq` | float | Compustat | Quarterly revenue ($M) |
| `epspxq` | float | Compustat | Basic EPS excluding extraordinary items |
| `cshoq` | float | Compustat | Common shares outstanding (M) |
| `prccq` | float | Compustat | Quarter-end closing price |
| `dlttq` | float | Compustat | Long-term debt ($M) |
| `actq` | float | Compustat | Current assets ($M) |
| `lctq` | float | Compustat | Current liabilities ($M) |

#### `ibes_forecasts.parquet`

| Column | Type | Source | Description |
|---|---|---|---|
| `ticker` | str | IBES | IBES ticker |
| `fpedats` | date | IBES | Forecast period end date |
| `fpi` | str | IBES | Forecast period indicator ("1"=next quarter) |
| `value` | float | IBES | Consensus analyst EPS forecast |
| `actual` | float | IBES | Actual reported EPS |
| `numest` | int | IBES | Number of estimates in consensus |
| `sue` | float | Derived | Standardized Unexpected Earnings (computed in `analyst_forecasts.py`) |

#### `ff_factors.parquet`

| Column | Type | Source | Description |
|---|---|---|---|
| `date` | date | Ken French | Month-end date |
| `mktrf` | float | Ken French | Market excess return (Mkt - Rf) |
| `smb` | float | Ken French | Small-minus-Big factor return |
| `hml` | float | Ken French | High-minus-Low (value) factor return |
| `rmw` | float | Ken French | Robust-minus-Weak (profitability) factor |
| `cma` | float | Ken French | Conservative-minus-Aggressive (investment) factor |
| `rf` | float | Ken French | Monthly risk-free rate |

#### `identifier_links.parquet`

| Column | Type | Description |
|---|---|---|
| `gvkey` | str | Compustat identifier |
| `permno` | int | CRSP identifier |
| `ticker` | str | Common ticker (for IBES link) |
| `link_dt` | date | Link effective start date |
| `linkenddt` | date | Link effective end date |
| `linktype` | str | WRDS link type (LC, LU, LS) |

### 5.2 Processed Data Files

#### `spacy_targets.parquet`

| Column | Type | Description |
|---|---|---|
| `transcript_id` | str | Source transcript |
| `firm_id` | str | Firm identifier (gvkey) |
| `quarter` | str | Quarter of earnings call |
| `target_text` | str | Full sentence containing the target |
| `target_type` | str | REVENUE / EPS / MARGIN / VOLUME / GROWTH / OTHER |
| `target_value` | float | Numeric value extracted (NaN if qualitative) |
| `target_unit` | str | billion / million / percent / other |
| `direction` | str | UP / DOWN / STABLE / UNCLEAR |
| `time_horizon` | str | NEXT_QUARTER / NEXT_YEAR / MULTI_YEAR / UNSPECIFIED |
| `confidence` | float | spaCy model confidence score (0–1) |

#### `llm_targets.parquet`

All columns from `spacy_targets.parquet` plus:

| Column | Type | Description |
|---|---|---|
| `hedge_level` | str | STRONG / MODERATE / WEAK / NONE — hedging language strength |
| `explicit_vs_implied` | str | EXPLICIT (numeric) or IMPLIED (directional only) |
| `cot_reasoning` | str | LLM chain-of-thought reasoning text |
| `model_used` | str | Which OpenAI model extracted this target |
| `prompt_strategy` | str | zero_shot / few_shot / chain_of_thought |
| `extraction_timestamp` | datetime | When the extraction was run |

#### `spacy_mt_scores.parquet` / `llm_mt_scores.parquet`

| Column | Type | Description |
|---|---|---|
| `gvkey` | str | Firm identifier |
| `quarter` | str | Quarter (current) |
| `mt_score` | float | Moving Targets score (positive = rising targets) |
| `n_targets` | int | Number of targets found this quarter |
| `n_matched` | int | Number of targets matched to prior quarter |
| `coverage_ratio` | float | `n_matched / n_targets` |
| `direction` | str | UP / DOWN / MIXED / FLAT |
| `mt_magnitude` | float | Absolute value of `mt_score` |

#### `panel_spacy.parquet` / `panel_llm.parquet` (merged panel)

| Column | Type | Description |
|---|---|---|
| `gvkey` | str | Firm identifier |
| `permno` | int | CRSP identifier |
| `quarter` | str | Current quarter |
| `ret_next_month` | float | CRSP return in the month after earnings call |
| `ret_ff_adj` | float | FF3-adjusted excess return |
| `mt_score` | float | MT signal (from spaCy or LLM) |
| `log_size` | float | Log market cap at quarter-end |
| `bm` | float | Book-to-Market ratio |
| `mom12` | float | 12-month momentum (ret_{t-12} to ret_{t-1}) |
| `sue` | float | Standardized Unexpected Earnings |
| `rev_growth` | float | Quarter-over-quarter revenue growth |

### 5.3 Derived Variable Formulas

#### Moving Targets (MT) Score
```
For each target type T where both Q and Q-1 have a numeric value:
  delta_T = (value_{T,Q} - value_{T,Q-1}) / max(|value_{T,Q-1}|, epsilon)
  weight_T = confidence_{T,Q} * coverage_T
  
MT_Q = Σ(weight_T * delta_T) / Σ(weight_T)
```
Where `epsilon = 1e-6` to avoid division by zero.

#### Book-to-Market (BM)
```
BM = ceqq / (prccq * cshoq)
```
Winsorized at 1st and 99th percentiles. Firms with negative book equity are excluded.

#### Momentum (MOM12)
```
MOM12_{t} = Π(1 + ret_m) for m in [t-12, t-2]  -  1
```
Note: Standard momentum skips the most recent month (t-1) to avoid short-term reversal.

#### Standardized Unexpected Earnings (SUE)
```
UE_t = actual_EPS_t - consensus_EPS_t
SUE_t = UE_t / std(UE_{t-8} to UE_{t-1})
```
Requires at least 4 quarters of history. Firms with fewer than 4 quarters are excluded.

#### Log Size
```
log_size = log(mktcap) where mktcap = |prc| * shrout (in $thousands)
```
Computed at the end of the month the earnings call falls in.

---

## 6. Evaluation Interpretation Guide

### 6.1 Fama-MacBeth Regression Results

**How to read the output table:**

```
Variable          Coef     Std Err   t-stat   p-value
mt_score         0.0042    0.0018    2.33     0.020   ← Key result
log_size        -0.0031    0.0009   -3.44     0.001   ← Size effect (expected negative)
bm               0.0028    0.0012    2.33     0.020   ← Value effect (expected positive)
mom12            0.0051    0.0016    3.19     0.002   ← Momentum (expected positive)
sue              0.0039    0.0011    3.55     0.000   ← PEAD (expected positive)
Intercept        0.0087    0.0021    4.14     0.000
```

**Interpreting the MT coefficient:**
- `mt_score = 0.0042` means a one-unit increase in MT is associated with 0.42% higher next-month return, all else equal
- **t-stat > 1.96** (or 2.0 by convention in finance): statistically significant at 5% level
- **t-stat > 2.58**: significant at 1% level — stronger result
- **t-stat < 1.5**: borderline, questionable significance

**What results would be "good":**
- MT coefficient is positive and t-stat > 2.0 (replicates the paper)
- LLM MT coefficient has a higher t-stat than spaCy MT (proves LLM extension adds value)
- Sign and approximate magnitude of control variables are economically sensible (size negative, BM positive, momentum positive, SUE positive)

**What results would be "concerning":**
- MT coefficient is negative: markets price in target revisions in the wrong direction — likely a data error
- All t-stats are very low (< 1.0): panel construction likely has a bug; check the time alignment
- MT t-stat > 5.0: suspiciously high — check for look-ahead bias in the merge

**Annualized alpha calculation:**
```
Monthly alpha = mt_coef × (cross-sectional std of mt_score)
Annualized alpha = monthly alpha × 12
```
Cohen & Nguyen (2024) find approximately 4–6% annualized alpha for a long-short portfolio.

### 6.2 NLP Evaluation Metrics

**Precision, Recall, F1:**
```
Precision = TP / (TP + FP)  — Of all extracted targets, what fraction are real?
Recall    = TP / (TP + FN)  — Of all real targets, what fraction did we find?
F1        = 2 × (P × R) / (P + R)  — Harmonic mean
```
Where TP/FP/FN are computed against the 100-segment manual annotation.

**Interpreting scores:**
- F1 > 0.80: Excellent extraction quality
- F1 0.65–0.80: Good — acceptable for financial NLP where targets are rare
- F1 < 0.65: Poor — check prompts, lower confidence threshold, expand entity rules

**Expected comparison:**
- spaCy baseline: F1 ≈ 0.55–0.70 (rule-based systems typically have high precision, lower recall)
- GPT-4o-mini (CoT): F1 ≈ 0.70–0.85 (LLMs find more implicit targets, improving recall)

**Inter-annotator Kappa interpretation:**
- κ < 0.40: Poor agreement — annotation guidelines need refinement
- κ 0.40–0.60: Moderate — acceptable for complex linguistic tasks
- κ > 0.60: Substantial — good annotation quality
- κ > 0.80: Almost perfect — publication-quality annotation

### 6.3 Portfolio Returns

**Calendar-time long-short portfolio:**
- Each month, sort all firms with MT scores into quintiles (Q1=lowest MT, Q5=highest MT)
- Long Q5, Short Q1
- Report monthly return and run a CAPM/FF3 time-series regression to get alpha

**Interpreting results:**
| Metric | Good | Acceptable | Concerning |
|---|---|---|---|
| Monthly alpha | > 0.4% | 0.2%–0.4% | < 0.1% or negative |
| t-stat (alpha) | > 2.5 | 1.5–2.5 | < 1.5 |
| Sharpe ratio (annualized) | > 0.5 | 0.2–0.5 | < 0.2 |
| Max drawdown | < 30% | 30%–50% | > 50% |

**Monotonicity check:** Verify that average returns increase from Q1 to Q5. If they don't (e.g., Q3 has the highest return), the MT signal may not be clean.

### 6.4 Ablation Study Interpretation

**What the ablation table tells you:**
- If CoT > few-shot > zero-shot: reasoning structure helps, and examples help
- If GPT-4o >> GPT-4o-mini: model capability matters, suggesting more complex language understanding is needed
- If RAG semantic > exact match: targets are paraphrased across quarters, not repeated verbatim
- If fine-tuned Mistral ≈ GPT-4o-mini: the fine-tuning successfully transferred knowledge at lower cost

**Reporting in the paper:** Lead with the main result (LLM > spaCy), then use the ablation to explain *why* — which component is driving the improvement.

---

## 7. Rubric Alignment Checklist

### 7.1 Proposal (10% — Already Submitted)

✅ Submitted — no action needed.

### 7.2 Presentation (30%)

| Rubric Item | Where It Appears | Status |
|---|---|---|
| Motivation and research question clearly stated | Slide 2–3 | Prepare slide content |
| System architecture explained | Slide 4 | Use 3-layer diagram |
| Technical depth: LLM techniques used | Slides 6–7 | CoT prompting, RAG, LangGraph |
| Results presented clearly | Slides 9–12 | Use tables from `comparison.py` |
| Live demo works | Slide 13 / demo tab | Pre-launch Gradio before class |
| Q&A handled competently | Preparation | Rehearse answers to 5 likely questions |
| Presentation within time limit | Full deck | Time the dry run |

**Likely Q&A questions to prepare for:**
1. "How did you validate that your LLM extraction is actually better?"
   - Answer: 100-segment manual annotation; F1 scores; FM t-stat comparison
2. "Why use LangGraph instead of a single LLM call?"
   - Answer: Separation of concerns; each agent can use different prompts/models; easier ablation; more interpretable intermediate state
3. "How much did the OpenAI API cost?"
   - Answer: Refer to `data/processed/llm_extraction_costs.json`; discuss cost-quality tradeoff
4. "Does this strategy actually make money, accounting for transaction costs?"
   - Answer: Long-short alpha is gross; realistic transaction costs for the relevant firm size would reduce but not eliminate alpha
5. "Why not just use the company's official guidance filings?"
   - Answer: Earnings calls contain more nuanced, forward-looking language; management may downplay formal guidance vs. verbal statements

### 7.3 Report (30%)

| Rubric Item | Report Section | Artifact |
|---|---|---|
| Problem motivation | Introduction | Written text |
| Related work cited | Related Work | Paper references |
| Data description | Data section | Table 1: summary stats |
| NLP methodology described | Methodology 4.1–4.3 | Architecture description |
| LLM techniques explained (prompting, RAG, agents) | Methodology 4.2–4.4 | Prompt examples in appendix |
| Empirical methodology | Methodology 4.5 | FM regression setup |
| NLP evaluation results | Results 5.1 | Table 2: F1 scores |
| Financial evaluation results | Results 5.2–5.3 | Tables 3–4 |
| Ablation analysis | Results 5.4 | Table 5 |
| Discussion of limitations | Discussion | Written text |
| Professional formatting | Throughout | LaTeX preferred |

**Generate all tables before starting to write:**
```bash
make tables   # See Makefile target — generates all LaTeX tables to docs/
```

**Report writing order (recommended):**
1. Write the Data section first (easy, factual)
2. Write Methodology from the codebase (describe what the code does)
3. Insert pre-generated tables into Results
4. Write Discussion and Limitations after seeing the results
5. Write Introduction last (summarize what you found)

### 7.4 Demo (20%)

| Rubric Item | Demo Feature | Status |
|---|---|---|
| Functional interactive demo | Both Gradio tabs | Test thoroughly |
| Shows NLP capabilities | Tab 1: Live extraction with agent trace | Pre-load demo text |
| Shows financial results | Tab 2: MT score chart, portfolio alpha | Pre-compute for demo firms |
| Handles errors gracefully | Error messages in UI | Test with bad inputs |
| Documentation for running demo | README demo section | Write clear instructions |

**Demo preparation checklist:**
- [ ] Gradio app launches in < 30 seconds
- [ ] Both tabs load without errors
- [ ] At least 3 demo firms have precomputed results
- [ ] A 3–5 sentence "example transcript snippet" is ready to paste into Tab 1
- [ ] The share URL is generated and tested on a different device
- [ ] Demo recording is saved at `demo/earningslens_demo.mp4`

### 7.5 GitHub Repository (10%)

| Rubric Item | Location | Checklist |
|---|---|---|
| Clear README | `README.md` | ☐ Install, quick-start, architecture, team |
| Well-organized code | `src/` structure | ☐ Consistent naming, no dead files |
| Meaningful commit history | `git log` | ☐ > 20 commits with descriptive messages |
| Requirements file | `requirements.txt` | ☐ All deps listed, pinned versions |
| No hardcoded credentials | All files | ☐ `git grep "sk-"` returns nothing |
| Tests present and passing | `tests/` | ☐ `pytest tests/ -v` all green |
| Documentation in code | All `.py` files | ☐ Docstrings on all public functions |
| `.gitignore` properly configured | `.gitignore` | ☐ Excludes data/, .env, .venv/ |

**Commit message conventions to use going forward:**
```
feat: Add semantic threshold calibration to SemanticContinuityMatcher
fix: Handle NaN MT scores in panel merge
docs: Update README with quick-start instructions
test: Add 5 edge-case tests for MovingTargetsComputer
refactor: Extract common WRDS query logic to base class
```

---

## 8. Task Division Recommendation

> This is a suggestion based on each person's likely strengths given the project structure. Modify freely based on actual availability and interest.

### Person A — Timothy Chan (tc3460)
**Focus: System architecture, data pipeline, LangGraph agents, demo**

| Task | Week | Deliverable |
|---|---|---|
| Set up shared repository, configure CI | Week 1, Day 1 | Working repo with `.gitignore`, `Makefile` |
| Run DataPipeline for all WRDS data | Week 1, Day 1–2 | All raw parquet files in `data/raw/` |
| Validate data and run `merge_all()` | Week 1, Day 2 | `panel_spacy.parquet` and `panel_llm.parquet` |
| Debug and extend LangGraph pipeline | Week 2, Day 5 | Full 4-agent pipeline working on subset |
| Build and test Gradio demo | Week 4, Day 1–2 | `demo/app.py` with precomputed results |
| Record demo video | Week 4, Day 2 | `demo/earningslens_demo.mp4` |
| Polish README and repo for submission | Week 5, Day 1–2 | Clean, documented repo |
| Prepare presentation slides | Week 5, Day 3 | Draft slides deck |
| Annotation: segments 000–032 | Week 1, Day 5 | 33 annotated segments |

**Why Timothy:** Data pipeline and agent orchestration require the most system integration; keeping them in one person's hands reduces coordination overhead.

### Person B — Yewen Li (yl5888)
**Focus: spaCy baseline, LLM extraction, fine-tuning, ablation studies**

| Task | Week | Deliverable |
|---|---|---|
| Run and validate spaCy baseline pipeline | Week 1, Day 3–4 | `spacy_targets.parquet`, `spacy_mt_scores.parquet` |
| Spot-check spaCy extractions | Week 1, Day 4 | `data/processed/spot_check_notes.txt` |
| Run LLM extraction pipeline (gpt-4o-mini) | Week 2, Day 1–2 | `llm_targets.parquet` |
| Run ablation: model and prompting strategy | Week 3, Day 3–4 | `data/processed/ablation/` |
| Run QLoRA fine-tuning on Colab (stretch) | Week 3, Day 5 | Fine-tuned Mistral model |
| Write Methodology sections 4.1–4.2 in report | Week 4, Day 3–5 | Report draft sections |
| Annotation: segments 033–065 | Week 1, Day 5 | 33 annotated segments |

**Why Yewen:** spaCy and LLM extraction are complementary — understanding the baseline informs how to design better LLM prompts. Ablations build directly on extraction.

### Person C — Tiantian Hang (th3166)
**Focus: RAG pipeline, evaluation, report writing, presentation**

| Task | Week | Deliverable |
|---|---|---|
| Build ChromaDB vector index | Week 2, Day 3 | ChromaDB persisted at `data/cache/chromadb` |
| Calibrate similarity threshold on annotation | Week 2, Day 4 | Updated `configs/config.yaml` |
| Run Fama-MacBeth regressions (both signals) | Week 3, Day 1–2 | `fm_results_spacy.csv`, `fm_results_llm.csv` |
| Run portfolio sorts and CAR analysis | Week 3, Day 2 | Portfolio alpha tables, CAR figure |
| Run ablation: RAG retrieval strategy | Week 3, Day 3–4 | RAG ablation results |
| Compile all evaluation results into report tables | Week 4, Day 3 | LaTeX tables in `docs/` |
| Write full report draft (lead author) | Week 4, Day 3–5 | Complete report draft |
| Prepare presentation introduction slides (slides 1–4) | Week 5, Day 3 | Slides 1–4 |
| Annotation: segments 066–099 | Week 1, Day 5 | 34 annotated segments |

**Why Tiantian:** RAG and evaluation require careful statistical thinking; combining them with report writing ensures the written interpretation matches the actual results produced.

### Cross-Team Coordination

**Daily standups (10 min):** Share blockers in the team Slack/Discord. Use a shared Google Doc to track what's been run and what outputs exist.

**Data handoffs:** Since all data lives in `data/`, the team should use a shared network drive or a service like Google Drive / Dropbox to share the large parquet files. Do not commit large data files to Git.

**API key usage:** One team member holds the OpenAI API key for extraction runs. Others can use the pre-computed outputs without needing API access.

**Merge conflicts:** Each person works in their own branch and opens PRs to `main`. Assign Timothy as the merge gatekeeper since he owns the system integration.

---

## 9. Troubleshooting FAQ

### 9.1 WRDS Connection Failures

**Problem:** `ConnectionRefusedError` or `wrds.Connection()` hangs indefinitely.

**Solutions:**
1. Verify your WRDS credentials are correct at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu) — use Columbia SSO
2. Check that you are on the Columbia network or VPN (WRDS may restrict off-campus access)
3. If using password in `.env`, ensure there are no trailing spaces or special characters
4. Try the interactive password prompt instead:
   ```python
   import wrds
   db = wrds.Connection(wrds_username="tc3460")
   ```
5. WRDS sometimes has maintenance windows — check [wrds-web.wharton.upenn.edu/wrds/support/](https://wrds-web.wharton.upenn.edu/wrds/support/) for status

**Problem:** `relation "ciq.wrds_transcript_detail" does not exist`

**Solution:** Request Capital IQ Transcripts dataset access:
1. Log in to WRDS → My Account → Subscription Manager
2. Request access to "Capital IQ" under the Transcripts section
3. Wait 1–2 business days for approval
4. Note: The table names may differ — check with `db.list_tables(library="ciq")`

**Problem:** WRDS query returns 0 rows for transcripts.

**Solutions:**
1. Check the date range — Capital IQ transcripts are available from roughly 2002 onward
2. Verify the universe filter — if `universe="sp500"`, ensure the S&P 500 constituent list is populated
3. Try a simpler query first:
   ```python
   db.raw_sql("SELECT * FROM ciq.wrds_transcript_detail LIMIT 5")
   ```

### 9.2 OpenAI API Rate Limits

**Problem:** `openai.error.RateLimitError: You have exceeded your rate limit.`

**Solutions:**
1. The `LLMTargetExtractor` has built-in exponential backoff — wait a few minutes and retry
2. Reduce `batch_size` in `configs/config.yaml` from 10 to 5
3. Add a delay between batches: set `configs.llm.batch_delay_seconds = 2`
4. Check your OpenAI usage at [platform.openai.com/usage](https://platform.openai.com/usage) — you may have hit your monthly spend limit
5. Switch to `gpt-4o-mini` if using `gpt-4o` — it has higher rate limits

**Problem:** `openai.error.AuthenticationError: No API key provided.`

**Solution:** Ensure `OPENAI_API_KEY` is set in your `.env` file AND that the config loader reads it:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

**Problem:** Extraction pipeline crashes midway, losing progress.

**Solution:** The pipeline uses `--checkpoint-every N` to save progress. Resume from checkpoint:
```bash
python -m src.llm_extraction.extraction_pipeline \
  --backend openai \
  --model gpt-4o-mini \
  --resume-from data/processed/llm_targets_checkpoint.parquet \
  --input data/raw/transcripts_raw.parquet \
  --output-dir data/processed
```

### 9.3 spaCy Model Not Found

**Problem:** `OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.`

**Solution:**
```bash
# Ensure you are in the virtual environment
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Download the model
python -m spacy download en_core_web_lg

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('OK')"
```

**Problem:** spaCy model is found but NER performance is poor.

**Solution:** Verify the large model (not small) is being used. `en_core_web_sm` has much weaker NER. Check `config.yaml`:
```yaml
baseline:
  spacy_model: "en_core_web_lg"  # NOT en_core_web_sm or en_core_web_md
```

### 9.4 ChromaDB Persistence Issues

**Problem:** `chromadb.errors.NotEnoughElementsException` or empty query results.

**Solutions:**
1. Verify the index was actually built: `store.collection.count()` should be > 0
2. Verify the persist directory exists and has data:
   ```bash
   ls -la data/cache/chromadb/
   # Should contain sqlite3.db and other files
   ```
3. If the index is corrupt, delete and rebuild:
   ```bash
   rm -rf data/cache/chromadb/
   python -c "from src.rag.vector_store import TargetVectorStore; ..."  # rebuild
   ```

**Problem:** `RuntimeError: Chroma requires sqlite >= 3.35.0` (common on older Linux/macOS).

**Solution:**
```bash
pip install pysqlite3-binary
```
Then add to the top of any script using ChromaDB:
```python
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

**Problem:** ChromaDB index is built but semantic queries return wrong results.

**Solution:** Verify the embedding model is consistent between indexing and querying. Both must use `text-embedding-3-small` (or whatever is set in `configs/config.yaml`). Mixing embedding models corrupts similarity scores.

### 9.5 LangGraph Import Errors

**Problem:** `ImportError: cannot import name 'StateGraph' from 'langgraph'`

**Solution:** LangGraph has frequent breaking API changes. Pin the version:
```bash
pip install "langgraph==0.1.5"  # Check requirements.txt for the pinned version
```

**Problem:** `TypeError: StateGraph.__init__() got an unexpected keyword argument`

**Solution:** Check `src/agents/graph.py` for the API version being used. If LangGraph was updated, the StateGraph constructor signature may have changed. Downgrade to the version in `requirements.txt`.

**Problem:** Agent graph runs but `state` doesn't have expected keys after invocation.

**Solution:** Check `src/agents/state.py` — the `PipelineState` TypedDict must have all keys that agents write. Missing keys cause silent `None` values. Run with `--log-level DEBUG` to see intermediate state.

### 9.6 Gradio Launch Issues

**Problem:** `gradio.exceptions.Error: Port 7860 is already in use.`

**Solution:**
```bash
# Kill whatever is using the port
lsof -ti:7860 | xargs kill -9
# Or use a different port
python demo/app.py --port 7861
```

**Problem:** Gradio `--share` URL doesn't work (connection times out).

**Solution:** Gradio share URLs use a Gradio cloud tunnel that can be unreliable. Alternatives:
- Use `ngrok` for more reliable tunneling: `ngrok http 7860`
- Run the demo locally and screen-share during the presentation

**Problem:** Demo Tab 1 (live extraction) times out after pasting a long transcript.

**Solution:** Long transcripts hit OpenAI's rate limits or the Gradio 60-second timeout. Pre-truncate input to the first 2,000 words:
```python
def truncate_input(text: str, max_words: int = 2000) -> str:
    return " ".join(text.split()[:max_words])
```

### 9.7 Memory Issues with Large Datasets

**Problem:** `MemoryError` when loading `transcripts_raw.parquet`.

**Solutions:**
1. Load only required columns:
   ```python
   df = pd.read_parquet("data/raw/transcripts_raw.parquet", columns=["transcript_id", "text", "quarter"])
   ```
2. Process in chunks:
   ```python
   import pyarrow.parquet as pq
   table = pq.read_table("data/raw/transcripts_raw.parquet")
   for batch in table.to_batches(max_chunksize=1000):
       df_chunk = batch.to_pandas()
       # process df_chunk
   ```
3. Reduce the universe: set `configs.data.universe = "sp500"` instead of `"all"`
4. Filter by date range before loading full data

**Problem:** ChromaDB index build runs out of memory.

**Solution:** Build the index in batches:
```python
store.build_index_batched(targets_df, batch_size=1000)
```
This method processes 1,000 targets at a time instead of loading all embeddings simultaneously.

**Problem:** spaCy baseline is very slow (> 2 hours for S&P 500).

**Solution:** Enable spaCy's batch processing and multiprocessing:
```python
# In baseline_pipeline.py or your script
texts = list(df['text'])
# Use nlp.pipe() instead of nlp() for efficiency
for doc in nlp.pipe(texts, batch_size=50, n_process=4):  # n_process = num CPU cores
    # process doc
```

---

## 10. Quick Reference Command Sheet

> Print this page and keep it handy. Commands are in execution order.

### Setup (One-Time)

| Step | Command | Notes |
|---|---|---|
| Create venv | `python3.10 -m venv .venv` | Once per machine |
| Activate venv | `source .venv/bin/activate` | Every session |
| Install deps | `pip install -r requirements.txt && pip install -e .` | Once per machine |
| Download spaCy | `python -m spacy download en_core_web_lg` | Once per machine |
| Copy env template | `cp .env.example .env` | Then edit with credentials |
| Run tests | `pytest tests/test_baseline.py -v` | Expect 39 passing |
| Verify WRDS | `python -c "import wrds; db = wrds.Connection(); print('OK')"` | Must succeed before data pull |

### Week 1: Data & Baseline

| Step | Command | Expected Output |
|---|---|---|
| Pull all WRDS data | `python -m src.data_retrieval.pipeline --start-year 2010 --end-year 2023 --universe sp500 --output-dir data/raw` | 7 parquet files in `data/raw/` |
| Run exploration | `python notebooks/01_data_exploration.py` | HTML report in `data/processed/exploration/` |
| Run spaCy baseline | `python -m src.baseline.baseline_pipeline --input data/raw/transcripts_raw.parquet --output-dir data/processed` | `spacy_targets.parquet`, `spacy_mt_scores.parquet` |
| Merge panel (spaCy) | `python -c "from src.data_retrieval.pipeline import DataPipeline; from src.utils.config_loader import load_config; p=DataPipeline(load_config()); p.merge_all('data/processed/spacy_mt_scores.parquet').to_parquet('data/processed/panel_spacy.parquet')"` | `panel_spacy.parquet` |

### Week 2: LLM & RAG

| Step | Command | Expected Output |
|---|---|---|
| Test LLM extraction (10 transcripts) | `python -m src.llm_extraction.extraction_pipeline --backend openai --model gpt-4o-mini --input data/raw/transcripts_raw.parquet --output-dir data/processed --limit 10` | `llm_targets.parquet` (10 rows) |
| Run full LLM extraction | `python -m src.llm_extraction.extraction_pipeline --backend openai --model gpt-4o-mini --input data/raw/transcripts_raw.parquet --output-dir data/processed --checkpoint-every 100` | `llm_targets.parquet` (full) |
| Build ChromaDB index | See Section 4.10 | `data/cache/chromadb/` populated |
| Run semantic MT | See Section 3, Week 2, Day 3 | `llm_mt_scores.parquet` |
| Merge panel (LLM) | Same as spaCy merge, but with `llm_mt_scores.parquet` | `panel_llm.parquet` |
| Run LangGraph subset | See Section 3, Week 2, Day 5 | `langgraph_subset_results.json` |

### Week 3: Evaluation

| Step | Command | Expected Output |
|---|---|---|
| FM regression (spaCy) | See Section 4.12 | `fm_results_spacy.csv` |
| FM regression (LLM) | See Section 4.12 | `fm_results_llm.csv` |
| Portfolio alpha | See Section 3, Week 3, Day 1 | `portfolio_alpha_*.csv` |
| CAR analysis | `python -m src.evaluation.announcement_cars --panel data/processed/panel_llm.parquet --daily-returns data/raw/crsp_daily_returns.parquet --output data/processed/car_results.parquet` | `car_results.parquet` |
| NLP metrics | `python -m src.evaluation.comparison --annotation-dir data/sample/annotation --predictions data/processed/spacy_targets.parquet data/processed/llm_targets.parquet --labels spaCy LLM --output data/processed/nlp_comparison.csv` | `nlp_comparison.csv` |
| Ablation table | See Section 3, Week 3, Day 3–4 | `data/processed/ablation_table.csv` |
| Generate LaTeX tables | `make tables` | `docs/table_*.tex` |

### Week 4: Demo & Report

| Step | Command | Expected Output |
|---|---|---|
| Pre-compute demo data | See Section 3, Week 4, Day 1 | `data/cache/llm_targets_demo.parquet` |
| Build demo ChromaDB | See Section 3, Week 4, Day 1 | `data/cache/chromadb_demo/` |
| Launch demo | `python demo/app.py --demo-mode precomputed --targets-path data/cache/llm_targets_demo.parquet --port 7860 --share` | Public Gradio URL |
| Record demo | QuickTime / OBS | `demo/earningslens_demo.mp4` |

### Week 5: Polish

| Step | Command | Expected Output |
|---|---|---|
| Format code | `black src/ demo/ --line-length 100` | Reformatted files |
| Lint | `flake8 src/ demo/ --max-line-length 100 --ignore E203,W503` | 0 errors |
| Final test run | `pytest tests/test_baseline.py -v` | 39 passing |
| Check for secrets | `git grep -r "sk-" -- "*.py" "*.yaml" "*.env"` | 0 results |
| Tag release | `git tag -a "final-submission" -m "STAT GR5293 Final Submission" && git push --tags` | Tag on remote |

### Makefile Shortcuts

```bash
make data        # Pull all WRDS data
make baseline    # Run spaCy baseline pipeline
make extract     # Run LLM extraction pipeline
make rag         # Build ChromaDB and run semantic matching
make eval        # Run all evaluation modules
make tables      # Generate all LaTeX tables
make demo        # Launch Gradio demo
make test        # Run test suite
make clean       # Remove __pycache__ and .pyc files (NOT data files)
make all         # Run entire pipeline end-to-end (use with caution)
```

---

## Appendix A: Project File Naming Conventions

All output files follow this naming convention to avoid confusion:

```
{source}_{content}_{version}.parquet

Examples:
  spacy_targets_v1.parquet      — spaCy-extracted targets, version 1
  llm_targets_gpt4o_cot.parquet — GPT-4o targets, chain-of-thought
  panel_llm_sp500.parquet        — Merged panel using LLM signal, S&P 500 universe
  fm_results_llm_ff3.csv         — FM regression results, LLM signal, FF3 controls
```

When running ablations, always use a `--suffix` argument to avoid overwriting production outputs.

---

## Appendix B: Key Paper Reference

Cohen, L., & Nguyen, H. (2024). Moving Targets: What Do Managers' Performance Targets Tell Us About Future Returns? *Journal of Finance*. [Available via Columbia library.]

**Key numbers from the paper to replicate:**
- MT long-short monthly alpha: ~0.42% (t-stat ~3.1) controlling for standard factors
- MT Fama-MacBeth coefficient: ~0.004 (one-unit MT → ~0.4% higher monthly return)
- Sample: S&P 500, 2005–2020
- Average targets per transcript: ~3.2

If your spaCy baseline produces numbers substantially different from these (e.g., alpha < 0.1% or t-stat < 1.0), investigate before proceeding to the LLM extension — the baseline must replicate before the extension can be meaningfully compared.

---

## Appendix C: Cost Budget

| Item | Estimated Cost | Who Pays |
|---|---|---|
| OpenAI GPT-4o-mini (full S&P 500 extraction) | $10–15 | Team shared |
| OpenAI GPT-4o (100-firm ablation subset) | $5–10 | Team shared |
| OpenAI embeddings (ChromaDB index) | $2–5 | Team shared |
| Google Colab Pro+ (fine-tuning) | $5–8 per run | Team shared |
| **Total** | **$25–40** | |

OpenAI costs can be monitored at [platform.openai.com/usage](https://platform.openai.com/usage). Set a spend alert at $30 to avoid surprises.

---

*End of EarningsLens Project Execution Guidebook.*  
*Questions? Post in the team channel. Good luck — ship it.*
