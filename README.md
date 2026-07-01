# Target Tracker

Research replication of the "Moving Targets" earnings-call commitment
continuity workflow. The runnable path is a single web app.

## Quick start

```bash
pip install -r requirements-app.txt   # first time only
python app.py
```

Equivalent via the package extra::

    pip install ".[app]"

That is the only command you need. On first launch the app:

- installs any missing Python packages
- creates the `data/` folder layout
- opens the Gradio UI at http://localhost:7860

## Workflow inside the app

1. **Download data manually** and place files under `data/raw/` (at minimum
   `ciq_transcripts.parquet`).
2. Open the **Pipeline** tab in the browser.
3. Run stages in order (or click **Run all**):
   - llm — LLM extraction (NB03, needs API key pasted in the app)
   - rag — semantic MT batch (NB04)
   - calibrate — threshold calibration (NB04b, needs labeled CSV in `data/processed/`)
   - cache — build analysis cache for the UI (NB06)
4. View results in **Entity Report** and **Watchlist**.

No terminal commands are required after `python app.py`.

## App tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | KPI dashboard and quick start |
| **Data** | Corpus upload and file checklist |
| **Pipeline** | Run every pipeline stage, view status and logs |
| **Entity Report** | Per-entity targets, risk gauge, dropped commitments |
| **Watchlist** | Ranked MT-risk screen for a quarter |

## Expected data files

Place manually downloaded files here:

| Path | Purpose |
|------|---------|
| `data/raw/ciq_transcripts.parquet` | Earnings call transcripts (required) |
| `data/processed/mt_calibration_sample_labeled.csv` | Human-labeled pairs for calibration (optional) |

The app writes intermediate outputs to `data/processed/` and the UI cache to
`data/cache/demo/`.

## Advanced: research stack (optional)

For WRDS data pulls, spaCy baseline replication (NB02), or evaluation
notebooks, install the research extras on top of the app runtime::

    pip install -r requirements-research.txt

Or selective extras::

    pip install ".[baseline]"   # spaCy NB02
    pip install ".[data]"       # WRDS NB01
    pip install ".[evaluation]" # statsmodels / scipy

The legacy spaCy baseline (`src/baseline/`, `scripts/run_spacy_baseline.py`)
is not part of the app workflow. Use it only for notebook replication.

See `notebooks/` and `docs/` for the original Colab workflow and methodology.
