"""
bootstrap.py — First-launch setup for ``python app.py``.

Users should only need one command::

    python app.py

This module installs missing Python packages, downloads the spaCy model, and
creates the expected ``data/`` folder layout. It runs once per process before
the Gradio UI is built.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False

# Packages the in-app workflow needs. Kept in requirements-app.txt as the
# canonical list; duplicated here so we can probe imports without parsing files.
_RUNTIME_IMPORTS = (
    ("gradio", "gradio"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("spacy", "spacy"),
    ("openai", "openai"),
    ("langgraph", "langgraph"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
)

SPACY_MODEL = "en_core_web_sm"


def _pip_install(requirements_path: Path) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],
        cwd=str(requirements_path.parent),
    )


def _missing_packages() -> List[str]:
    missing: List[str] = []
    for module_name, pip_name in _RUNTIME_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def _ensure_spacy_model() -> Optional[str]:
    try:
        import spacy
    except ImportError:
        return "spaCy is not installed yet."

    try:
        spacy.load(SPACY_MODEL)
        return None
    except OSError:
        pass

    logger.info("Downloading spaCy model %s (first launch only)…", SPACY_MODEL)
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
    )
    return f"Downloaded spaCy model `{SPACY_MODEL}`."


def _ensure_data_layout(project_root: Path) -> None:
    for sub in ("raw", "processed", "cache/demo", "cache/chromadb_experiment"):
        (project_root / "data" / sub).mkdir(parents=True, exist_ok=True)


def ensure_ready(project_root: Optional[Path] = None) -> List[str]:
    """
    Bootstrap the app environment. Safe to call multiple times.

    Returns human-readable status lines suitable for the Workflow tab.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return []

    project_root = project_root or Path(__file__).resolve().parent.parent
    messages: List[str] = []

    req_file = project_root / "requirements-app.txt"
    missing = _missing_packages()
    if missing:
        messages.append(
            "Installing missing packages (first launch only): "
            + ", ".join(missing)
        )
        if not req_file.exists():
            raise FileNotFoundError(f"Missing requirements file: {req_file}")
        _pip_install(req_file)
        messages.append("Python dependencies installed.")

    _ensure_data_layout(project_root)
    messages.append(
        "Data folders ready under `data/raw/`, `data/processed/`, `data/cache/demo/`."
    )

    spacy_note = _ensure_spacy_model()
    if spacy_note:
        messages.append(spacy_note)

    _BOOTSTRAPPED = True
    if not messages:
        messages.append("Environment ready.")
    return messages
