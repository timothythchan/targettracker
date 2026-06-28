#!/usr/bin/env python3
"""Top-level entry point for the EarningsLens app.

This is the shortest path to launching the web UI:

    python app.py
    python app.py --host 0.0.0.0 --port 7860
    python app.py --share

It is a thin wrapper around the unified ``earningslens`` CLI's ``app``
subcommand, which itself wraps ``demo.cli.main``. For the full set of
pipeline subcommands, use::

    python -m src --help
    python -m src status
    python -m src baseline --limit 20
    python -m src cache
    python -m src app --port 7860

If no demo cache exists under ``data/cache/demo/``, the app still
launches but shows a "Demo cache not built yet" banner with the
subcommand to run::

    python -m src cache         # NB06 port
    python -m src pipeline      # full pipeline, end to end
"""

from __future__ import annotations

import sys

from demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
