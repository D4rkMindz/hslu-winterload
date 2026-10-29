"""CLI entry point: `uv run python -m winterload.pipeline` runs every feature pipeline for the
proposal's default date range; pass -s/-e to backfill a different range.
"""

from __future__ import annotations

import datetime as dt

from . import run_all

if __name__ == "__main__":
    run_all()
