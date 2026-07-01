#!/usr/bin/env python3
"""Launch the Target Tracker web app.

This is the only command you need::

    python app.py

On first launch the app installs any missing Python packages, creates the
``data/`` folder layout, and opens the Gradio UI. Place your manually
downloaded files under ``data/raw/``, then run each pipeline stage from the
**Pipeline** tab inside the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from demo.bootstrap import ensure_ready

ensure_ready(_ROOT)

from demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
