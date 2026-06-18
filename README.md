# Moving Targets LM / EarningsLens

This repository contains a scriptable replication and LLM extension of the
"Moving Targets" earnings-call target extraction workflow. The original work is
still documented in `docs/` and exploratory notebooks remain in `notebooks/`,
but the runnable path is now normal Python modules and scripts — and a Gradio
**web app** you can launch with a single command.

## Run as an app (no Colab, no WRDS subscription)

The repo ships with a small bundled sample cache (`demo/sample_cache/`) so the
Gradio UI works on a fresh clone with no external data. Pick whichever launcher
matches your environment:

### Option A — Python (recommended for development)

```bash
python -m pip install -r requirements-app.txt   # ~minimal deps
python app.py                                   # http://localhost:7860
```

`requirements-app.txt` contains only what the app needs to render the cached
results (`gradio`, `pandas`, `pyarrow`, `jinja2`). Use `requirements.txt`
instead if you want the full research pipeline (spaCy, LangGraph, WRDS, etc.).

### Option B — Make

```bash
make install   # installs requirements-app.txt
make app       # launches on http://127.0.0.1:7860
```

### Option C — Docker

```bash
docker build -t earningslens-app .
docker run --rm -p 7860:7860 earningslens-app
```

Then open <http://localhost:7860>.

The app opens with two tabs:

- **Company Analysis** — per-ticker, per-quarter target table, dropped-target
  table, risk gauge, and spaCy-vs-LLM comparison.
- **Portfolio Screen** — ranked Moving-Targets risk score across the universe
  for a selected quarter.

When the bundled sample cache is in use the UI shows a "Sample data mode"
banner so illustrative numbers are never confused with real research output.

### Building the real demo cache

To replace the bundled sample with real, end-to-end pipeline output you need a
WRDS subscription and the full deps:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m spacy download en_core_web_lg

python scripts/run_data_retrieval.py --wrds-user YOUR_WRDS_USERNAME --output-dir data/raw
python scripts/run_spacy_baseline.py --input data/raw/transcripts.parquet
python scripts/run_llm_extraction.py --backend openai --model gpt-4o-mini \
    --input data/raw/transcripts.parquet --output-dir data/processed
# Notebook 08 materialises data/cache/demo/{pipeline_cache.json,portfolio_screen.json}.
```

Once `data/cache/demo/pipeline_cache.json` exists the app picks it up
automatically on next launch and drops the sample-mode banner.

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

### 4. Launch the local app

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
