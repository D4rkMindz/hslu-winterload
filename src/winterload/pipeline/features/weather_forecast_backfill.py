"""Backfills weather_forecast_stats with what was really known at 08:00 Zurich time on D-1,
reconstructed from archived ECMWF IFS HRES single runs -- see docs/proposal.pdf, reviewer point A1.

Writes one Parquet file per day under weather_forecast/ (a Feast FileSource can read a directory
of Parquet files as one dataset) so a long backfill checkpoints as it goes: an interrupted run
loses at most one day of API calls, and rerunning the same command resumes by skipping any day
that already has a file.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from winterload import COORDINATES
from winterload.config import settings
from winterload.pipeline.features import OpenMeteoBackfillPipeline
from winterload.source.openmeteo import HOURLY_VARIABLES, OpenMeteo

logger = logging.getLogger(__name__)

ZURICH_TZ = "Europe/Zurich"

# ECMWF IFS HRES needs ~4-6h after initialisation before Open-Meteo has disseminated it, so the
# nominal "08:00 Zurich" run (06:00 UTC) would not actually be published by 08:00 Zurich time.
# The most recent run that genuinely was available by then is the previous 00:00 UTC run --
# which puts the real lead time at ~22-45h rather than the proposal's nominal 16-39h.
RUN_HOUR_UTC = 0


class WeatherForecastBackfill(OpenMeteoBackfillPipeline):
    name = "weather_forecast_stats"

    @staticmethod
    def get_forecast_run(target_day: dt.date) -> dt.datetime:
        """The ECMWF run that was actually available at 08:00 Zurich time on `day_minus_1`."""
        return dt.datetime.combine(target_day - dt.timedelta(days=1), dt.time(RUN_HOUR_UTC, 0))

    def day_path(self, target_day: dt.date) -> str:
        return f"{settings.feast_store_path}/weather_forecast/{target_day.isoformat()}.parquet"

    def load_day(self, client: OpenMeteo, target_day: dt.date, cities: list[dict] = COORDINATES) -> list[dict]:
        """One row per (city, hour) for the 24 local hours of `target_day`."""
        run = self.get_forecast_run(target_day)
        rows = {}

        for city in cities:
            rows[city["name"]] = client.get_weather_backfill(city["lat"], city["lon"], run=run)

        return rows

    def transform_day(self, target_day: dt.date, rows: list[dict]) -> pd.DataFrame:
        # Built from the calendar date, not `local_start + pd.Timedelta(days=1)`: Timedelta is a fixed
        # 24h physical duration, which overshoots/undershoots by an hour on DST transition days (23h in
        # spring, 25h in autumn) and causes an hour to be dropped or double-counted between two days.
        local_start = pd.Timestamp(target_day, tz=ZURICH_TZ)
        local_end = pd.Timestamp(target_day + dt.timedelta(days=1), tz=ZURICH_TZ)

        df = pd.DataFrame(rows["hourly"])
        day_hours = df.loc[(df["date"] >= local_start) & (df["date"] < local_end)].copy()
        day_hours["run_timestamp"] = pd.Timestamp(self.get_forecast_run(target_day), tz="UTC")
        day_hours["lead_time_hours"] = (day_hours["date"] - day_hours["run_timestamp"]).dt.total_seconds() / 3600

        return day_hours

    def feature_columns(self) -> list[str]:
        return ["event_timestamp", "location_id", *HOURLY_VARIABLES, "run_timestamp", "lead_time_hours"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-e", "--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days that already have a file")
    args = parser.parse_args()

    written, skipped, failed = WeatherForecastBackfill().build_range(args.start, args.end, force=args.force)
    logger.info(
        "Done: wrote %s new day(s), skipped %s already-backfilled day(s), "
        "rejected %s day(s) after repeated failures",
        written, skipped, failed,
    )
