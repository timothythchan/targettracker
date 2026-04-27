# MovingTargetsLM

> LLM-powered extraction of management tone and forward guidance from earnings call transcripts for alpha generation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

EarningsLens is a research pipeline that extracts structured signals — **management tone**, **quantitative targets**, and **forward guidance** — from S&P 500 earnings call transcripts using a combination of rule-based NLP (spaCy) and large language models (GPT-4o). Extracted signals are stored in a RAG vector store and routed through a LangGraph multi-agent pipeline to produce interpretable risk reports and Fama-MacBeth alpha estimates.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │         Raw Earnings Call Transcripts         │
                        │   (WRDS / SEC EDGAR, S&P 500, 2018–2024)    │
                        └───────────────────┬─────────────────────────┘
                                            │
                          ┌─────────────────▼─────────────────┐
                          │         Preprocessing Layer         │
                          │  segment split · speaker diarize   │
                          └──────────┬──────────────┬──────────┘
                                     │              │
                        ┌────────────▼───┐  ┌───────▼────────────┐
                        │ spaCy Baseline │  │   LLM Extractor    │
                        │  (rule-based)  │  │   (GPT-4o / FT)    │
                        │  MT · Tone     │  │  Targets · Hedging │
                        └────────────┬───┘  └───────┬────────────┘
                                     │              │
                          ┌──────────▼──────────────▼──────────┐
                          │        RAG Vector Store             │
                          │  ChromaDB · sentence-transformers   │
                          └───────────────┬─────────────────────┘
                                          │
                          ┌───────────────▼─────────────────────┐
                          │       LangGraph Multi-Agent          │
                          │  Orchestrator → Analyst → Validator  │
                          └──────────┬──────────────┬───────────┘
                                     │              │
                        ┌────────────▼───┐  ┌───────▼────────────┐
                        │  Risk Report   │  │ Fama-MacBeth Alpha  │
                        │  (HTML / PDF)  │  │  (panel regression) │
                        └────────────────┘  └────────────────────┘
```

---

## Features

- **Dual-extractor design** — compare a fast spaCy rule-based baseline against GPT-4o zero-shot and fine-tuned extraction
- **Management Tone (MT) scoring** — net positive/negative sentiment ratio following Cohen & Nguyen (2024)
- **Quantitative target extraction** — revenue, EPS, margin, and capex forward guidance with uncertainty hedging scores
- **RAG-augmented context** — retrieve similar historical transcripts at inference time for few-shot grounding
- **LangGraph orchestration** — stateful multi-agent pipeline with automated validation and retry logic
- **Fama-MacBeth cross-sectional regression** — test whether tone/guidance signals predict next-quarter abnormal returns
- **Interactive Gradio demo** — upload a transcript, get a full risk report in seconds

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/earningslens.git
cd earningslens
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

Or install only runtime dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in WRDS_USERNAME and OPENAI_API_KEY
```

### 4. Download the spaCy model

```bash
make setup
# Equivalent to: python -m spacy download en_core_web_lg
```

### 5. Run the data pipeline

```bash
make data
# Downloads transcripts from WRDS, preprocesses, and builds the vector store
```

### 6. Run analysis

```bash
# spaCy baseline extraction
make baseline

# LLM extraction (requires OPENAI_API_KEY)
make llm

# Fama-MacBeth evaluation
make evaluate
```

### 7. Launch the demo

```bash
make demo
# Opens Gradio UI at http://localhost:7860
```

---

## Project Structure

```
earningslens/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── .env.example
├── .gitignore
│
├── src/
│   └── earningslens/
│       ├── __init__.py
│       ├── config.py               # Central config loader (YAML + .env)
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── wrds_loader.py      # WRDS connection & transcript download
│       │   ├── preprocessing.py    # Segment splitting, speaker diarization
│       │   └── schema.py           # Pydantic data models
│       │
│       ├── extractors/
│       │   ├── __init__.py
│       │   ├── baseline.py         # spaCy rule-based extractor (MT score)
│       │   ├── llm_extractor.py    # GPT-4o zero-shot extractor
│       │   └── fine_tuned.py       # LoRA fine-tuned extractor (optional)
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── embedder.py         # sentence-transformers embedding
│       │   └── retriever.py        # ChromaDB vector store interface
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── graph.py            # LangGraph state graph definition
│       │   ├── agents.py           # Orchestrator, Analyst, Validator agents
│       │   └── run.py              # CLI entry point
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── fama_macbeth.py     # Cross-sectional panel regression
│       │   ├── metrics.py          # Extraction quality metrics (F1, MAE)
│       │   └── report.py           # HTML/PDF report generator
│       │
│       └── demo/
│           ├── __init__.py
│           └── app.py              # Gradio demo application
│
├── notebooks/
│   └── 01_data_exploration.py      # Data exploration (percent format)
│
├── tests/
│   ├── __init__.py
│   ├── fixtures/                   # Small sample transcripts for tests
│   └── test_baseline.py
│
├── data/
│   ├── raw/                        # Downloaded transcripts (gitignored)
│   ├── processed/                  # Parquet files (gitignored)
│   └── embeddings/                 # ChromaDB persist dir (gitignored)
│
└── outputs/
    ├── reports/                    # Generated risk reports
    └── figures/                    # Evaluation plots
```

---

## Data Requirements

EarningsLens requires access to **Wharton Research Data Services (WRDS)**:

| Dataset | Table | Description |
|---------|-------|-------------|
| Refinitiv / StreetEvents | `tr_ds_equitymds.wrds_transcript` | Earnings call transcripts |
| Compustat | `comp.funda` | Annual/quarterly fundamentals |
| CRSP | `crsp.dsf` | Daily stock returns |
| IBES | `ibes.statsumu` | Analyst consensus EPS estimates |

Request WRDS access at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu/). Columbia students: access is available through the library.

---

## Usage Examples

### spaCy Baseline Extraction

```python
from earningslens.extractors.baseline import SpacyTargetExtractor

extractor = SpacyTargetExtractor()
result = extractor.extract(transcript_text)

print(result.mt_score)          # e.g. 0.142
print(result.positive_words)    # ["growth", "strong", "exceeded"]
print(result.negative_words)    # ["uncertainty", "headwinds"]
```

### LLM Extraction

```python
from earningslens.extractors.llm_extractor import LLMExtractor

extractor = LLMExtractor(model="gpt-4o")
result = extractor.extract(transcript_text, ticker="AAPL", quarter="2024Q3")

print(result.revenue_guidance)      # {"low": 89.0, "high": 91.0, "unit": "B"}
print(result.eps_guidance)          # {"value": 1.55, "vs_consensus": +0.05}
print(result.hedging_score)         # 0.31  (0 = certain, 1 = maximally hedged)
print(result.overall_tone)          # "cautiously optimistic"
```

### RAG Retrieval

```python
from earningslens.rag.retriever import TranscriptRetriever

retriever = TranscriptRetriever()
similar = retriever.retrieve(
    query="supply chain disruption impacting margins",
    n_results=5,
    filter={"sector": "Technology"}
)
for doc in similar:
    print(doc.ticker, doc.quarter, doc.distance)
```

### Full LangGraph Pipeline

```python
from earningslens.pipeline.run import run_pipeline

report = run_pipeline(
    ticker="NVDA",
    quarter="2024Q4",
    transcript_path="data/raw/NVDA_2024Q4.txt"
)
report.save("outputs/reports/NVDA_2024Q4_report.html")
```

### Fama-MacBeth Regression

```python
from earningslens.evaluation.fama_macbeth import FamaMacBeth

fm = FamaMacBeth(
    signals=["mt_score", "hedging_score", "revenue_surprise"],
    controls=["log_me", "bm", "mom12m"]
)
results = fm.fit(panel_df)
results.summary()
```

---

## Evaluation

| Method | Signal | IC (mean) | t-stat | Ann. Alpha |
|--------|--------|-----------|--------|------------|
| spaCy Baseline | MT Score | 0.021 | 1.84 | — |
| GPT-4o Zero-Shot | MT + Targets | 0.038 | 3.21 | 4.2% |
| GPT-4o Fine-Tuned | MT + Targets | 0.051 | 4.07 | 6.1% |

*Results are illustrative; actual numbers will be filled after full pipeline runs.*

Evaluation methodology follows Cohen & Nguyen (2024): Fama-MacBeth cross-sectional regressions of next-quarter buy-and-hold abnormal return (BHAR) on extracted signals, controlling for log market equity, book-to-market, and 12-month momentum.

---

## Demo

```bash
make demo
```

Opens a Gradio interface at `http://localhost:7860` where you can:

1. Paste or upload an earnings call transcript
2. Select extraction method (spaCy / GPT-4o / Fine-tuned)
3. View extracted targets, tone score, and hedging score
4. Download a full risk report as PDF

---

## Team

| Name | UNI |
|------|-----|
| Timothy Chan |
| Yewen Li |
| Tiantian Hang |

Columbia University · Spring 2026

---

## References

- Cohen, L., & Nguyen, H. (2024). *Lazy Prices: New Evidence from Earnings Calls*. Working Paper.
- Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? *Journal of Finance*, 66(1), 35–65.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Fama, E., & MacBeth, J. (1973). Risk, Return, and Equilibrium. *Journal of Political Economy*, 81(3), 607–636.
- LangGraph: https://github.com/langchain-ai/langgraph

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

*EarningsLens is a research prototype. Nothing here constitutes investment advice.*
