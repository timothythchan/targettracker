"""Allow ``python -m src`` as an alias for the unified ``earningslens`` CLI."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
