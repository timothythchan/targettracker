# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

EarningsLens (MovingTargetsLM) is a Jupyter-notebook-based academic research pipeline (Columbia, STAT GR5293). It extracts investment signals from S&P 500 earnings call transcripts using spaCy NLP and GPT-4o, then evaluates them via Fama-MacBeth regressions. All code lives in 8 notebooks under `Notebooks/`; there is no `src/` directory in the repo (notebooks reference `src.*` modules that exist only on the original authors' machines).

### Key services

| Service | Port | Command |
|---------|------|---------|
| JupyterLab | 8888 | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=""` |
| Gradio demo | 7860 | Run the demo cells in `Notebooks/08_demo_preparation.ipynb`, or launch a standalone script |

### Important caveats

- **No `src/` directory**: The notebooks import from `src.baseline.*`, `src.llm_extraction.*`, `src.rag.*`, `src.agents.*`, and `src.evaluation.*`, but these modules are not committed to the repo. Running those notebook cells will fail with `ModuleNotFoundError`. All other cells (data exploration, inline code, visualizations) work fine.
- **No `requirements.txt` or `pyproject.toml`**: Dependencies are inferred from notebook imports. The update script installs all needed packages.
- **External API keys required**: Notebooks 03–08 require `OPENAI_API_KEY` for LLM extraction. Notebook 01 requires WRDS institutional credentials (`WRDS_USERNAME`). Set these as environment variables or in a `.env` file at the project root.
- **WRDS requires interactive password**: The `wrds` Python package prompts for a password interactively. For non-interactive use, create `~/.pgpass` with `wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD`. A `WRDS_PASSWORD` secret is also needed if automating connection.
- **No automated tests**: There is no `tests/` directory or test framework. Validation is done by running notebook cells.
- **Data is gitignored**: The `data/` directory with parquet files must be regenerated via WRDS (notebook 01). Without WRDS credentials, data-dependent cells will not run.
- **ChromaDB runs in-process**: No separate server needed; it persists to `data/cache/chromadb/`.
- **Python 3.12 is used in the cloud environment**; the original authors used 3.10–3.14. All dependencies are compatible.

### Linting / Testing

There is no linter configuration or test suite in this repo. To validate the environment, run a quick Python import check:

```bash
python3 -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('OK')"
```

### Running notebooks

Start JupyterLab and open notebooks from the `Notebooks/` directory. Notebooks are numbered 01–08 and should be run in order, though each can be explored independently for cells that don't depend on upstream data.
