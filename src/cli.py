"""
cli.py — Unified ``earningslens`` command-line interface.

Replaces six separate ``scripts/run_*.py`` invocations with one
discoverable entry point::

    earningslens --help

    earningslens status        # inspect data/ and report pipeline state
    earningslens data          # NB01 - WRDS data pull
    earningslens baseline      # NB02 - spaCy + Moving Targets
    earningslens llm           # NB03 - LLM target extraction
    earningslens rag           # NB04 - semantic MT batch
    earningslens calibrate     # NB04b - threshold calibration
    earningslens cache         # NB06 - build Gradio demo cache
    earningslens pipeline      # full chain
    earningslens app           # launch Gradio app

Each subcommand forwards its remaining arguments to the existing stage
``main(argv)`` function, so anything documented in ``--help`` on the
per-stage script still works after the rename, e.g.::

    earningslens baseline --limit 100
    earningslens llm --backend openai --model gpt-4o-mini
    earningslens app --port 7860 --share

The standalone ``scripts/run_*.py`` files are kept as thin wrappers so
existing automation (Makefile, CI) does not break.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("earningslens.cli")


# ---------------------------------------------------------------------------
# Subcommand registry
# ---------------------------------------------------------------------------

# Each entry: (subcommand, module dotted path, attribute, short description).
# The attribute must be a callable ``main(argv: Optional[List[str]]) -> int``
# compatible with the stage's own argparse layer.
_STAGES: List[tuple] = [
    (
        "data",
        "src.data_retrieval.cli",
        "main",
        "WRDS data retrieval — populates data/raw/ (NB01)",
    ),
    (
        "baseline",
        "src.baseline.baseline_pipeline",
        "main",
        "spaCy Moving Targets baseline — writes spacy_targets/spacy_mt_scores (NB02)",
    ),
    (
        "llm",
        "src.llm_extraction.extraction_pipeline",
        "main",
        "LLM target extraction — writes llm_targets.parquet/.jsonl (NB03)",
    ),
    (
        "rag",
        "scripts.run_rag_matching",
        "main",
        "Semantic MT batch via ChromaDB — writes semantic_mt_scores/* (NB04)",
    ),
    (
        "calibrate",
        "scripts.run_threshold_calibration",
        "main",
        "Threshold calibration — writes mt_calibration_result.json (NB04b)",
    ),
    (
        "cache",
        "scripts.build_demo_cache",
        "main",
        "Build the Gradio demo cache — writes data/cache/demo/* (NB06)",
    ),
    (
        "pipeline",
        "scripts.run_pipeline",
        "main",
        "Run every stage in order, each in its own subprocess",
    ),
    (
        "app",
        "demo.cli",
        "main",
        "Launch the Gradio web app",
    ),
]


def _load_stage_main(module_path: str, attr: str) -> Callable[..., int]:
    """Lazy import the stage main() to avoid pulling heavy deps for --help."""
    module = importlib.import_module(module_path)
    main = getattr(module, attr, None)
    if main is None:
        raise ImportError(f"{module_path}.{attr} is not defined")
    return main


# ---------------------------------------------------------------------------
# Subcommands implemented inside cli.py
# ---------------------------------------------------------------------------

def _status_main(argv: Optional[Sequence[str]] = None) -> int:
    """``earningslens status`` — print pipeline-state summary."""
    parser = argparse.ArgumentParser(
        prog="earningslens status",
        description="Inspect data/ and report which pipeline stages have run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Project data root (default: ./data).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text table.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from .status import describe_pipeline_status, status_dict

    if args.json:
        import json
        print(json.dumps(status_dict(args.data_dir), indent=2, default=str))
    else:
        print(describe_pipeline_status(args.data_dir))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_top_parser() -> argparse.ArgumentParser:
    """Top-level parser. Stage args are passed through verbatim via REMAINDER."""
    parser = argparse.ArgumentParser(
        prog="earningslens",
        description=(
            "EarningsLens / Moving Targets LM — unified command-line app.\n\n"
            "Every subcommand forwards extra args to the underlying stage; run "
            "`earningslens <subcommand> --help` for the per-stage flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    # status is implemented inline so it never imports heavy deps.
    sub_status = sub.add_parser(
        "status",
        help="Inspect data/ and report pipeline state.",
        add_help=False,
    )
    sub_status.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )

    # Stage subcommands: just hold a REMAINDER bucket; we hand-off to the
    # underlying main().
    for name, _mod, _attr, help_text in _STAGES:
        sp = sub.add_parser(name, help=help_text, add_help=False)
        sp.add_argument(
            "rest", nargs=argparse.REMAINDER,
            help=argparse.SUPPRESS,
        )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``earningslens`` entry point. Returns a process exit code."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if not raw_argv or raw_argv[0] in ("-h", "--help"):
        _build_top_parser().print_help()
        return 0

    subcommand = raw_argv[0]
    rest = raw_argv[1:]

    if subcommand in ("-h", "--help"):
        _build_top_parser().print_help()
        return 0

    if subcommand == "status":
        return _status_main(rest)

    # Stage dispatch
    for name, module_path, attr, _help_text in _STAGES:
        if subcommand != name:
            continue
        try:
            stage_main = _load_stage_main(module_path, attr)
        except ImportError as exc:
            print(
                f"earningslens: cannot load stage '{subcommand}' from "
                f"{module_path}.{attr}: {exc}",
                file=sys.stderr,
            )
            return 2
        try:
            result = stage_main(rest)
            return int(result) if result is not None else 0
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else (1 if exc.code else 0)

    # Unknown subcommand
    print(f"earningslens: unknown subcommand '{subcommand}'\n", file=sys.stderr)
    _build_top_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
