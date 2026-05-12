# Moving Targets LM / EarningsLens

This repository contains a scriptable replication and LLM extension of the
"Moving Targets" earnings-call target extraction workflow. The original work is
still documented in `docs/` and exploratory notebooks remain in `notebooks/`,
but the runnable path is now normal Python modules and scripts.

## Can this run without Google Colab?

Yes. The core pipeline can run on any machine with Python 3.10+:

1. WRDS data retrieval writes raw parquet files under `data/raw/`.
2. The spaCy baseline reads `data/raw/ciq_transcripts.parquet` and writes baseline
   targets and Moving Targets scores under `data/processed/`.
3. The LLM extractor reads the same transcript data and writes LLM targets under
   `data/processed/`.
4. The Gradio demo can be launched locally from Python.

Google Colab is only useful for optional GPU-heavy QLoRA fine-tuning. It is not
required for data retrieval, baseline extraction, hosted-API LLM extraction, RAG,
evaluation, or the demo.

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
  --input data/raw/ciq_transcripts.parquet \
  --output-dir data/processed
```

This writes:

- `data/processed/spacy_targets.parquet`
- `data/processed/spacy_mt_scores.parquet`

Equivalent installed console command:

```bash
earningslens-baseline --input data/raw/ciq_transcripts.parquet --output-dir data/processed
```

### 3. Run LLM extraction

Smoke test on a small subset:

```bash
python scripts/run_llm_extraction.py \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/ciq_transcripts.parquet \
  --output-dir data/processed \
  --limit 10
```

Full run:

```bash
python scripts/run_llm_extraction.py \
  --backend openai \
  --model gpt-4o-mini \
  --input data/raw/ciq_transcripts.parquet \
  --output-dir data/processed \
  --max-concurrent 10
```

This writes `data/processed/llm_targets.parquet` and
`data/processed/llm_extraction_summary.json`.

Equivalent installed console command:

```bash
earningslens-llm --backend openai --model gpt-4o-mini --input data/raw/ciq_transcripts.parquet --output-dir data/processed --limit 10
```

### 4. Launch the local demo

```bash
python scripts/run_demo.py --host 127.0.0.1 --port 7860
```

Equivalent installed console command:

```bash
earningslens-demo --host 127.0.0.1 --port 7860
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
