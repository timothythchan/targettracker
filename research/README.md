# Research-only modules

These packages are **not** used by `python app.py`. Install extras first::

    pip install -r requirements-research.txt

| Path | Purpose | Entry point |
|------|---------|-------------|
| `src/baseline/` | spaCy NB02 replication | `pip install ".[baseline]"` then `earningslens-baseline` |
| `src/data_retrieval/` | WRDS NB01 data pull | `pip install ".[data]"` then `earningslens-data` |
| `src/evaluation/` | Fama-MacBeth, CARs, portfolio sorts | `pip install ".[evaluation]"` |
| `src/llm_extraction/fine_tuning.py` | QLoRA fine-tuning | `pip install ".[finetune]"` |
| `scripts/run_spacy_baseline.py` | NB02 wrapper | see above |

The app pipeline is **llm → rag → calibrate → cache** only.
