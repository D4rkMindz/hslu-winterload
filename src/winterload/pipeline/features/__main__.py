"""CLI entry point: `uv run python -m winterload.pipeline` runs every feature pipeline for the
proposal's default date range; pass -s/-e to backfill a different range.
"""

from __future__ import annotations

import datetime as dt

from . import DEFAULT_END_DATE, DEFAULT_START_DATE, run_all

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--start", type=dt.date.fromisoformat, default=DEFAULT_START_DATE,
        help=f"default: {DEFAULT_START_DATE.isoformat()} (start of the forecast archive, docs/proposal.pdf)",
    )
    parser.add_argument(
        "-e", "--end", type=dt.date.fromisoformat, default=DEFAULT_END_DATE,
        help=f"default: {DEFAULT_END_DATE.isoformat()} (end of the held-out test year, docs/proposal.pdf)",
    )
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    run_all(args.start, args.end, force=args.force)
