# Moving Targets LM / EarningsLens

Research replication of the "Moving Targets" earnings-call target extraction
workflow. Original notebooks remain in `notebooks/` for reference; the
runnable path is a single web app.

## Quick start

```bash
python app.py
```

That is the only command you need. On first launch the app:

- installs any missing Python packages
- downloads the spaCy language model (`en_core_web_sm`)
- creates the `data/` folder layout
- opens the Gradio UI at http://localhost:7860

## Workflow inside the app

1. **Download data manually** and place files under `data/raw/` (at minimum
   `ciq_transcripts.parquet`).
2. Open the **Workflow** tab in the browser.
3. Run stages in order (or click **Run all**):
   - baseline — spaCy targets + Moving Targets (NB02)
   - llm — LLM extraction (NB03, needs API key pasted in the app)
   - rag — semantic MT batch (NB04)
   - calibrate — threshold calibration (NB04b, needs labeled CSV in `data/processed/`)
   - cache — build analysis cache for the UI (NB06)
4. View results in **Company Analysis** and **Portfolio Screen**.

No terminal commands are required after `python app.py`. The optional
`python -m src` CLI still exists for automation, but the app is the
primary interface.

## App tabs

| Tab | Purpose |
|-----|---------|
| **Workflow** | Run every pipeline stage, view status and logs |
| **Company Analysis** | Per-ticker targets, risk gauge, spaCy vs LLM comparison |
| **Portfolio Screen** | Ranked portfolio view for a quarter |

## Expected data files

Place manually downloaded files here:

| Path | Purpose |
|------|---------|
| `data/raw/ciq_transcripts.parquet` | Earnings call transcripts (required) |
| `data/processed/mt_calibration_sample_labeled.csv` | Human-labeled pairs for calibration (optional) |

The app writes intermediate outputs to `data/processed/` and the UI cache to
`data/cache/demo/`.

## Docker

```bash
make docker-build
make docker-run    # http://localhost:7860
```

## Advanced: CLI (optional)

For scripting and CI, the same stages are available via `python -m src`:

```bash
python -m src status
python -m src baseline --limit 20
python -m src cache
```

See `notebooks/` and `docs/` for the original Colab workflow and methodology.
