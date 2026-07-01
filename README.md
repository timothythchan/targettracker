# Target Tracker

Research replication of the "Moving Targets" earnings-call commitment
continuity workflow. Original notebooks remain in `notebooks/` for reference;
the runnable path is a single web app.

## Quick start

```bash
python app.py
```

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

## Advanced: research baseline (optional)

The legacy spaCy baseline pipeline (`src/baseline/`, NB02) remains in the repo
for notebook replication and academic comparison, but it is not part of the app
workflow or runtime dependencies. Use `scripts/run_spacy_baseline.py` directly
if you need those artifacts.

See `notebooks/` and `docs/` for the original Colab workflow and methodology.
