"""Backfills energy_load_stats with realised Swiss system load from ENTSO-E, one Parquet file
per day under energy_load/ -- same resumable layout as weather_forecast_backfill.py, so a long backfill
checkpoints as it goes and rerunning the same command resumes by skipping any day that already
has a file.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
# Must run before `fsspec` is imported by anything below (incl. transitively via feast/dask):
# fsspec snapshots its env-var config (FSSPEC_S3_*) at first import, so loading .env any later
# leaves the SeaweedFS S3 credentials unset and historical retrieval fails with NoCredentialsError.
from winterload.config import settings  # noqa: F401
from winterload.pipeline.features import BackfillPipeline, retry
from winterload.source.entso_e import EntsoE

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["event_timestamp", "bidding_zone_id", "quantity", "unit"]


class EnergyLoadBackfill(BackfillPipeline):
    name = "energy_load_stats"
    table = "energy_load"

    def load_day(self, target_day: dt.date, client: EntsoE):
        start = dt.datetime.combine(target_day, dt.time.min)
        end = dt.datetime.combine(target_day + dt.timedelta(days=1), dt.time.min) - dt.timedelta(seconds=1)
        return client.get_load(start_date=start, end_date=end)

    def transform_day(self, rows) -> pd.DataFrame:
        """One row per load measurement ENTSO-E reports for `target_day`."""
        result = pd.DataFrame(rows).rename(
            columns={"timestamp": "event_timestamp", "bidding_zone": "bidding_zone_id"}
        )
        return result[FEATURE_COLUMNS]

    def build_range(self, start_date: dt.date, end_date: dt.date, force: bool = False) -> tuple[int, int, int]:
        client = EntsoE()
        written = skipped = failed = 0
        day = start_date
        while day <= end_date:
            logger.info("%s: processing %s (%s to %s)", self.name, day, start_date, end_date)
            exists = self.day_exists(day)
            if not force and exists:
                skipped += 1
                day += dt.timedelta(days=1)
                continue
            if retry(
                    lambda:
                    self.write_day(
                        self.transform_day(
                            self.load_day(day, client)
                        ),
                        day),
                    f"Backfilling {day}"):
                written += 1
            else:
                failed += 1
            day += dt.timedelta(days=1)
        return written, skipped, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-e", "--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    written, skipped, failed = EnergyLoadBackfill().build_range(args.start, args.end, force=args.force)
    logger.info(
        "Done: wrote %s new day(s), skipped %s already-backfilled day(s), "
        "rejected %s day(s) after repeated failures",
        written, skipped, failed,
    )
