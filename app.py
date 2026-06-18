#!/usr/bin/env python3
"""Top-level entry point for the EarningsLens Gradio app.

This makes the repository runnable as a self-contained app with a single
command:

    python app.py

It is a thin wrapper around `demo.cli.main`. Flags are passed through, so
the following all work:

    python app.py
    python app.py --host 0.0.0.0 --port 7860
    python app.py --share

When no real demo cache has been pre-computed under `data/cache/demo/`, the
app falls back to the bundled illustrative sample cache under
`demo/sample_cache/`, so a fresh clone can launch the UI without any
external data sources or notebook execution.
"""

from __future__ import annotations

import sys

from demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
