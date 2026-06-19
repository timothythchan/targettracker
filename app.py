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

If no demo cache exists under `data/cache/demo/`, the app still launches
but shows a "Demo cache not built yet" banner with the script to run:

    python scripts/build_demo_cache.py     # NB06 port
    python scripts/run_pipeline.py         # full pipeline, end to end
"""

from __future__ import annotations

import sys

from demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
