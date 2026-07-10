"""
pipeline_runner.py — Run pipeline stages in-process for the Gradio Pipeline tab.

The web UI calls stage ``main()`` functions directly instead of shelling out
to ``python -m src …``, so users never need to touch the CLI.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import queue
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (stage id, label, default extra argv for the in-app buttons)
WORKFLOW_STAGES: List[Tuple[str, str, str]] = [
    ("llm",       "LLM target extraction (NB03)",             "--limit 50"),
    ("rag",       "Semantic MT batch (NB04)",                ""),
    ("calibrate", "Threshold calibration (NB04b)",           ""),
    ("cache",     "Build analysis cache (NB06)",             ""),
    ("all",       "Run all stages in order",                 ""),
]

_STAGE_LOADERS: Dict[str, Tuple[str, str]] = {
    "llm":       ("src.llm_extraction.extraction_pipeline", "main"),
    "rag":       ("scripts.run_rag_matching", "main"),
    "calibrate": ("scripts.run_threshold_calibration", "main"),
    "cache":     ("scripts.build_demo_cache", "main"),
    "all":       ("scripts.run_pipeline", "main"),
}


def _load_main(stage: str) -> Callable[..., int]:
    if stage not in _STAGE_LOADERS:
        raise ValueError(f"Unknown stage: {stage}")
    module_path, attr = _STAGE_LOADERS[stage]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    module = importlib.import_module(module_path)
    main = getattr(module, attr, None)
    if main is None:
        raise ImportError(f"{module_path}.{attr} is not defined")
    return main


def _resolve_raw_transcript_path(data_dir: Path) -> Optional[Path]:
    raw_dir = data_dir / "raw"
    for name in ("ciq_transcripts.parquet", "transcripts.parquet"):
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def _build_argv(stage: str, extra_args: str, data_dir: Path) -> List[str]:
    """Merge smart defaults with user-supplied flags."""
    argv: List[str] = []
    try:
        argv.extend(shlex.split(extra_args or ""))
    except ValueError as exc:
        raise ValueError(f"Could not parse extra args: {exc}") from exc

    if stage == "all":
        # User downloads data manually — skip optional WRDS pull by default.
        if not any(a == "--skip" for a in argv):
            argv = ["--skip", "data", *argv]
        if "--start" not in argv:
            argv = ["--start", "llm", *argv]

    return argv


def run_stage_streaming(
    stage: str,
    extra_args: str = "",
    *,
    data_dir: Optional[Path] = None,
    api_key: str = "",
) -> Generator[str, None, None]:
    """
    Run one workflow stage and yield growing log text for a Gradio Textbox.

    Parameters
    ----------
    stage:
        One of the ids in :data:`WORKFLOW_STAGES`.
    extra_args:
        Additional argv tokens, e.g. ``--limit 20``.
    data_dir:
        Project data root (default: ``<repo>/data``).
    api_key:
        Optional LLM API key forwarded to ``OPENAI_API_KEY`` for this run.
    """
    if not stage:
        yield "Select a stage first."
        return

    data_dir = data_dir or (PROJECT_ROOT / "data")
    log_queue: queue.Queue[Optional[str]] = queue.Queue()

    # Forward API key for llm / cache stages without requiring shell exports.
    prior_key = None
    if api_key and api_key.strip():
        import os

        prior_key = (
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("EARNINGSLENS_LLM_API_KEY"),
        )
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        os.environ["EARNINGSLENS_LLM_API_KEY"] = api_key.strip()

    try:
        argv = _build_argv(stage, extra_args, data_dir)
    except ValueError as exc:
        yield str(exc)
        return

    header = f"▶ {stage}  {' '.join(argv)}\n\n"
    yield header

    class _QueueHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_queue.put(self.format(record) + "\n")
            except Exception:
                self.handleError(record)

    def _worker() -> None:
        root = logging.getLogger()
        handler = _QueueHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.setLevel(logging.DEBUG)
        prior_handlers = list(root.handlers)
        prior_level = root.level
        root.handlers = [handler]
        root.setLevel(logging.INFO)

        stdout_capture = io.StringIO()
        exit_code = 1
        try:
            main_fn = _load_main(stage)
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stdout_capture
            ):
                exit_code = int(main_fn(argv) or 0)
            captured = stdout_capture.getvalue()
            if captured:
                log_queue.put(captured)
        except Exception as exc:
            log_queue.put(f"\nERROR: {type(exc).__name__}: {exc}\n")
            exit_code = 1
        finally:
            log_queue.put(f"\n[finished — exit code {exit_code}]\n")
            root.handlers = prior_handlers
            root.setLevel(prior_level)
            log_queue.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    accumulated = header
    while True:
        try:
            chunk = log_queue.get(timeout=0.15)
        except queue.Empty:
            if not thread.is_alive():
                while True:
                    try:
                        leftover = log_queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is None:
                        break
                    accumulated += leftover
                    yield accumulated
                break
            yield accumulated
            continue

        if chunk is None:
            break
        accumulated += chunk
        yield accumulated

    # Restore env if we temporarily set an API key.
    if prior_key is not None:
        import os

        for env_name, value in (
            ("OPENAI_API_KEY", prior_key[0]),
            ("EARNINGSLENS_LLM_API_KEY", prior_key[1]),
        ):
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = value

    # Small pause so filesystem writes are visible to the status refresh.
    time.sleep(0.2)
