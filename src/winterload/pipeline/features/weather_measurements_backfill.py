"""Backfills weather_measurements_stats with Open-Meteo's observed (short-lead) weather per city,
one Parquet file per day under weather_measurements/ -- same resumable layout as
weather_forecast_backfill.py, so a long backfill checkpoints as it goes and rerunning the same command
resumes by skipping any day that already has a file.
"""

import datetime as dt
import logging

import pandas as pd

from winterload import COORDINATES, Coordinate
from winterload.pipeline.features import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.pipeline.features.openmeteo import OpenMeteoBackfillPipeline, DayResult
from winterload.source.openmeteo import HOURLY_VARIABLES, OpenMeteo, OpenMeteoModelResult

logger = logging.getLogger(__name__)


class WeatherMeasurementsBackfill(OpenMeteoBackfillPipeline):
    name = "weather_measurements_stats"
    table = "weather_measurements"

    def load_day(self, client: OpenMeteo, target_day: dt.date,
                 cities: list[Coordinate] = COORDINATES) -> DayResult:
        rows = {}

        for city in cities:
            rows[city["name"]] = client.get_weather_historic(city["lat"], city["lon"],
                                                             start_date=target_day, end_date=target_day)

        return rows

    def transform_day(self, target_day: dt.date, openmeteo_result: OpenMeteoModelResult) -> pd.DataFrame:
        return pd.DataFrame(openmeteo_result["hourly"])

    def feature_columns(self) -> list[str]:
        return ["event_timestamp", "location_id", *HOURLY_VARIABLES]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--start", type=dt.date.fromisoformat, default=DEFAULT_START_DATE,
        help=f"default: {DEFAULT_START_DATE.isoformat()}",
    )
    parser.add_argument(
        "-e", "--end", type=dt.date.fromisoformat, default=DEFAULT_END_DATE,
        help=f"default: {DEFAULT_END_DATE.isoformat()}",
    )
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    written, skipped, failed = WeatherMeasurementsBackfill(logger=logger).run(
        start_date=args.start, end_date=args.end, force=args.force)
    logger.info(
        "Done: wrote %s new day(s), skipped %s already-backfilled day(s), "
        "rejected %s day(s) after repeated failures",
        written, skipped, failed,
    )


if __name__ == "__main__":
    main()
