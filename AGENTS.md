# AGENTS.md

## Cursor Cloud specific instructions

Target Tracker (a.k.a. EarningsLens / Moving Targets LM) is a single Python product: a
Gradio web app that extracts forward-looking "targets" from earnings-call transcripts
(spaCy baseline + Moving Targets scoring, optional LLM/RAG/LangGraph stages). The UI is
an institutional dashboard with tabs **Overview, Data, Pipeline, Entity Report,
Watchlist** (built from `demo/interface.py` + `demo/theme.py`). There is **no separate
backend, database, or queue** — ChromaDB is embedded and WRDS/LLM providers are
remote and optional. Standard commands live in `README.md` and the `Makefile`; the
notes below are the non-obvious bits.

### Running the app
- Run with `python app.py` (or `make app`). The dev startup script already installs
  deps and the spaCy model, so on a warm VM you can launch directly.
- `app.py` calls `demo/bootstrap.py:ensure_ready()` first; it pip-installs anything
  missing and downloads the spaCy `en_core_web_sm` model on first launch (needs
  internet that one time).
- Default bind is `0.0.0.0:7860` (`demo/cli.py`). Use `--host 127.0.0.1 --port 7860`
  for local-only.
- Gradio is v6.x here; it prints a harmless `UserWarning` that `theme`/`css` moved
  from the `Blocks` constructor to `launch()`. Ignore it — the app still serves.

### Data is required to see results (and is gitignored)
- The pipeline reads `data/raw/ciq_transcripts.parquet`. The whole `data/` tree is
  gitignored, and the real source (WRDS / Capital IQ) is an **external, paid/academic
  service**, so fresh data is not available out of the box.
- For local development/testing without WRDS, drop in a small synthetic parquet. The
  baseline loader accepts normalized columns `companyid, fiscalyear, fiscalquarter,
  component_type, text` (or the raw CIQ names `componenttext, component_type_id, year,
  quarter`). Put forward-looking sentences in `text` (MONEY/PERCENT/PRODUCT entities,
  e.g. "Net income was $1.2 billion", "we expect a 12% increase in iPhone sales") so
  spaCy actually emits targets.
- **Moving Targets scoring needs history:** MT measures compare a quarter against
  t-4, so each company needs ≥5 consecutive quarters or `compute_mt` logs
  "insufficient t-4 history" and writes no `spacy_mt_scores.parquet` (target
  extraction still works).

### Which stages need secrets
- **baseline** stage (Pipeline tab → "Run stage") runs the full spaCy extraction +
  Moving Targets entirely offline — no API key. This is the easiest end-to-end check.
- **llm**, **cache**, and the LangGraph **agents**/Entity-Report live path need an
  LLM key via env (`EARNINGSLENS_LLM_API_KEY`, falling back to `GOOGLE_API_KEY` /
  `GEMINI_API_KEY` / `OPENAI_API_KEY`) or pasted into the app's "LLM API key" field.
  The `cache` stage also expects the curated CIQ demo universe (AAPL, MSFT, NVDA,
  META, GOOGL, T) and the full CIQ schema, so it will not run on arbitrary
  synthetic tickers.
- The **Entity Report** and **Watchlist** tabs render from a prebuilt cache
  under `data/cache/demo/`. With no cache they fall back to the live LangGraph
  pipeline (LLM key required); otherwise they show "No data available".

### Tests / lint / build
- There is **no automated test suite and no linter/formatter configured** (no
  `tests/`, no pytest/ruff/black/mypy config; `make clean` only deletes caches).
  Don't expect `pytest`/`make lint` to exist.
- CLI mirror of the app for automation: `python -m src status`,
  `python -m src baseline --raw-path data/raw/ciq_transcripts.parquet`,
  `python -m src cache`, etc.
- There is no Docker support (the `Dockerfile`/`make docker-*` targets were removed);
  run the app directly with `python app.py`.
