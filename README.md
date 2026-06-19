# Moving Targets LM / EarningsLens

This repository contains a scriptable replication and LLM extension of the
"Moving Targets" earnings-call target extraction workflow. The original work is
still documented in `docs/` and exploratory notebooks remain in `notebooks/`,
but the runnable path is now normal Python modules and scripts — and a Gradio
**web app** you can launch with a single command.

## Run as an app

The Gradio UI is the end of the pipeline. Once the cache it reads
(`data/cache/demo/pipeline_cache.json` + `portfolio_screen.json`) has been
built by `scripts/build_demo_cache.py`, the app is a single command:

```bash
python -m pip install -r requirements-app.txt    # gradio + pandas + pyarrow
python app.py                                    # http://localhost:7860
# Or:
make app
docker run --rm -p 7860:7860 earningslens-app
```

If the cache has not been built yet the app still launches but shows a
prominent "Demo cache not built yet" banner explaining which script to run.
The repo no longer ships any synthetic stub data — what you see in the UI
is always real pipeline output.

The app has two tabs:

- **Company Analysis** — per-ticker / per-quarter targets, dropped-target
  table, risk gauge, and spaCy-vs-LLM comparison.
- **Portfolio Screen** — ranked Moving-Targets risk score across the demo
  universe for a selected quarter.

## Notebook-free pipeline

Every step that used to live in a Colab notebook has a corresponding
Python script under `scripts/`. The mapping is:

| Notebook                              | Script                                       | What it does                                                                                                  |
| ------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `01_data_retrieval_v2.ipynb`          | `scripts/run_data_retrieval.py`              | Pull CRSP / Compustat / IBES / FF / CIQ from WRDS into `data/raw/`                                            |
| `02_spacy_baseline_v2.ipynb`          | `scripts/run_spacy_baseline.py`              | spaCy targets + Moving Targets baseline → `spacy_targets.parquet`, `spacy_mt_scores.parquet`                  |
| `03_llm_extraction_v2.ipynb`          | `scripts/run_llm_extraction.py`              | LLM target extraction (Gemini / OpenAI) → `llm_targets.parquet` (+ optional `llm_targets.jsonl` resumable flow) |
| `04_rag_matching_v4.ipynb`            | `scripts/run_rag_matching.py`                | Semantic MT via ChromaDB + sentence-transformers → `semantic_mt_scores.parquet`, `per_pair_sims.parquet`      |
| `04b_threshold_calibration.ipynb`     | `scripts/run_threshold_calibration.py`       | F1 sweep + logistic + bootstrap CI → `mt_calibration_result.json`                                             |
| `05_langgraph_agents_v3.ipynb`        | _(library only — `src/agents/`)_              | Defines the 4-agent LangGraph pipeline used by the cache builder + Gradio app                                  |
| `06_demo_preparation_v2.ipynb`        | `scripts/build_demo_cache.py`                | Build `data/cache/demo/{pipeline_cache,portfolio_screen,spacy_results,llm_results}.json`                      |

The notebooks are kept for narrative + diagnostic plots, but everything
they materialise is now produced by these scripts. Run them in sequence
with the orchestrator:

```bash
python scripts/run_pipeline.py                          # everything end-to-end
python scripts/run_pipeline.py --start rag              # reuse llm_targets.parquet
python scripts/run_pipeline.py --skip data llm          # reuse upstream parquets
python scripts/run_pipeline.py --dry-run                # print the plan only
```

Each stage runs as a separate subprocess so a failure in one stage does
not poison the next.

### What you need before running each stage

- **`data`** stage: a WRDS subscription. Configure as you would for a
  notebook (`~/.pgpass` or `--wrds-user YOUR_USERNAME`).
- **`baseline`** stage: a spaCy model. `python -m spacy download en_core_web_lg`.
- **`llm`** stage: an API key in `OPENAI_API_KEY` / `GOOGLE_API_KEY` /
  `GEMINI_API_KEY`. Use `--use-jsonl-flow` for resumable runs.
- **`rag`** stage: `pip install chromadb sentence-transformers` (the heavy
  RAG deps). GPU is optional via `--device cuda`.
- **`calibrate`** stage: `pip install scikit-learn` and a labeled
  `data/processed/mt_calibration_sample_labeled.csv`.
- **`demo`** stage: all of the above outputs already on disk.

Once `data/cache/demo/pipeline_cache.json` is materialised, the Gradio app
picks it up on the next launch and the "Demo cache not built yet" banner
disappears.

## Can this run without Google Colab?

Yes. Everything above runs on any machine with Python 3.10+:

1. WRDS data retrieval writes raw parquet files under `data/raw/`.
2. The spaCy baseline reads `data/raw/transcripts.parquet` and writes baseline
   targets and Moving Targets scores under `data/processed/`.
3. The LLM extractor reads the same transcript data and writes LLM targets under
   `data/processed/`.
4. The Gradio app can be launched locally from Python or in Docker (see above).

Google Colab is only useful for optional GPU-heavy QLoRA fine-tuning. It is not
required for data retrieval, baseline extraction, hosted-API LLM extraction, RAG,
evaluation, or the app.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m spacy download en_core_web_lg
```

For OpenAI-backed LLM extraction:

```bash
export OPENAI_API_KEY="..."
```

For WRDS retrieval, configure WRDS credentials as you normally would (for
example `~/.pgpass`) or pass `--wrds-user`.

## Notebook-free commands

### 1. Pull data from WRDS

```bash
python scripts/run_data_retrieval.py \
  --wrds-user YOUR_WRDS_USERNAME \
  --output-dir data/raw
```

Equivalent installed console command:

```bash
earningslens-data --wrds-user YOUR_WRDS_USERNAME --output-dir data/raw
```

### 2. Run the spaCy baseline

```bash
python scripts/run_spacy_baseline.py \
  --input data/raw/transcripts.parquet \
```

### 3. Run LLM extraction

Smoke test on a small subset:

```bash
python scripts/run_llm_extraction.py \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/transcripts.parquet \
  --output-dir data/processed \
  --limit 10
```

Full run:

```bash
python scripts/run_llm_extraction.py \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/transcripts.parquet \
  --output-dir data/processed \
  --max-concurrent 10
```

This writes `data/processed/llm_targets.parquet` and
`data/processed/llm_extraction_summary.json`.

Equivalent installed console command:

```bash
earningslens-llm --backend openai --model gpt-4o-mini --input data/raw/transcripts.parquet --output-dir data/processed --limit 10
```

### 4. Build the demo cache (replaces NB06)

```bash
python scripts/build_demo_cache.py
```

This reads everything under `data/raw/` + `data/processed/` and writes
`data/cache/demo/pipeline_cache.json`, `portfolio_screen.json`,
`spacy_results.json`, and `llm_results.json` — the four files the Gradio
app reads.

If `llm_targets.parquet` was produced by an older NB03 run with the
trailing-zero company_id truncation bug, pass `--repair-llm-parquet` to
rebuild it from the canonical JSONL.

### 5. Launch the local app

```bash
python app.py --host 127.0.0.1 --port 7860
```

Equivalent options:

```bash
python scripts/run_demo.py --host 127.0.0.1 --port 7860
earningslens-demo --host 127.0.0.1 --port 7860
make app HOST=127.0.0.1 PORT=7860
docker run --rm -p 7860:7860 earningslens-app
```

## Transcript input expectations

The baseline pipeline expects a parquet file with at least:

- `companyid`
- `fiscalyear`
- `fiscalquarter`
- `component_type`
- `text`

The LLM pipeline accepts either:

- a directory containing `transcripts.parquet` or `ciq_transcripts.parquet`,
- a direct parquet file path via `--input`, or
- a directory of JSON transcript documents.

For parquet inputs, the loaders accept both notebook-style columns (`transcript_id`, `text`, `component_type`) and the WRDS CIQ retrieval output (`transcriptid`, `componenttext`, `component_type_id`, `year`, `quarter`). If no transcript ID exists, the LLM loader creates transcript groups from available company-quarter columns.

## Repository hygiene

Generated data, caches, virtual environments, and bytecode are ignored by
`.gitignore`. Keep large parquet files in local storage or shared drives rather
than committing them.
