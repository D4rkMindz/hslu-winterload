"""Backfills weather_measurements_stats with Open-Meteo's observed (short-lead) weather per city,
one Parquet file per day under weather_measurements/ -- same resumable layout as
weather_forecast_backfill.py, so a long backfill checkpoints as it goes and rerunning the same command
resumes by skipping any day that already has a file.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from winterload import COORDINATES
from winterload.pipeline.features import OpenMeteoBackfillPipeline
from winterload.source.openmeteo import HOURLY_VARIABLES, OpenMeteo

logger = logging.getLogger(__name__)


class WeatherMeasurementsBackfill(OpenMeteoBackfillPipeline):
    name = "weather_measurements_stats"
    table = "weather_measurements"

    def load_day(self, client: OpenMeteo, target_day: dt.date, cities: list[dict] = COORDINATES) -> pd.DataFrame:
        rows = {}

        for city in cities:
            rows[city["name"]] = client.get_weather_historic(city["lat"], city["lon"],
                                                     start_date=target_day, end_date=target_day)

        return rows

    def transform_day(self, target_day: dt.date, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows["hourly"])

    def feature_columns(self) -> list[str]:
        return ["event_timestamp", "location_id", *HOURLY_VARIABLES]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-e", "--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    written, skipped, failed = WeatherMeasurementsBackfill().build_range(args.start, args.end, force=args.force)
    logger.info(
        "Done: wrote %s new day(s), skipped %s already-backfilled day(s), "
        "rejected %s day(s) after repeated failures",
        written, skipped, failed,
    )
