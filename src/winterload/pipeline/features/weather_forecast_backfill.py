"""Backfills weather_forecast_stats with what was really known at 08:00 Zurich time on D-1,
reconstructed from archived ECMWF IFS HRES single runs. The run rule lives in
winterload.forecast.features.dataset.weather_run, reviewer point A1.

Writes one Parquet file per day under weather_forecast/ (a Feast FileSource can read a directory
of Parquet files as one dataset) so a long backfill checkpoints as it goes: an interrupted run
loses at most one day of API calls, and rerunning the same command resumes by skipping any day
that already has a file.
"""
import datetime as dt

import pandas as pd

from winterload import COORDINATES, Coordinate
from winterload.forecast.features.dataset import FIRST_WEATHER_DAY, local_day_bounds, weather_run
from winterload.pipeline.features import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.pipeline.features.openmeteo import OpenMeteoBackfillPipeline
from winterload.source.openmeteo import HOURLY_VARIABLES, OpenMeteo, OpenMeteoResult, OpenMeteoModelResult


class WeatherForecastBackfill(OpenMeteoBackfillPipeline):
    name = "weather_forecast_stats"
    table = "weather_forecast"

    def run(self, start_date: dt.date, end_date: dt.date,
            force: bool = False, cities: list[Coordinate] = COORDINATES) -> None:
        # The archive rejects earlier runs, and winterload.pipeline.features.retry waits 60 seconds
        # for each rejected day. Starting at the default 2022-01-01 would idle for over 13 hours.
        super().run(max(start_date, FIRST_WEATHER_DAY), end_date, force, cities)

    def load_day(self, client: OpenMeteo, target_day: dt.date,
                 cities: list[Coordinate] = COORDINATES) -> dict[str, OpenMeteoResult]:
        """One row per (city, hour) for the local hours of `target_day`."""
        run = weather_run(target_day)
        rows = {}

        for city in cities:
            rows[city["name"]] = client.get_weather_backfill(city["lat"], city["lon"], run=run)

        return rows

    def transform_day(self, target_day: dt.date, openmeteo_result: OpenMeteoModelResult) -> pd.DataFrame:
        local_start, local_end = local_day_bounds(target_day)
        df = pd.DataFrame(openmeteo_result["hourly"])
        day_hours = df.loc[(df["date"] >= local_start) & (df["date"] < local_end)].copy()
        day_hours["run_timestamp"] = pd.Timestamp(weather_run(target_day), tz="UTC")
        day_hours["lead_time_hours"] = (day_hours["date"] - day_hours["run_timestamp"]).dt.total_seconds() / 3600

        return day_hours

    def feature_columns(self) -> list[str]:
        return ["event_timestamp", "location_id", "model", *HOURLY_VARIABLES, "run_timestamp", "lead_time_hours"]


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

    WeatherForecastBackfill().run(
        start_date=args.start, end_date=args.end, force=args.force)


if __name__ == "__main__":
    main()
