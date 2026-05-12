"""Lightweight CLI for launching the EarningsLens Gradio demo.

This module avoids importing gradio/pandas at import time so ``--help`` works
before demo dependencies are installed.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the demo argument parser without importing the web app."""
    parser = argparse.ArgumentParser(description="Launch EarningsLens Gradio demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the demo and return a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        from .app import build_interface
    except ImportError as exc:
        raise SystemExit(
            "demo dependencies are missing. Install them with "
            "`python -m pip install -r requirements.txt` before launching the demo. "
            f"Original error: {exc}"
        ) from exc

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
