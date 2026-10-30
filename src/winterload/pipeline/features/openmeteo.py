import datetime as dt
from abc import abstractmethod

import pandas as pd

from winterload import COORDINATES, Coordinate
from winterload.pipeline.features import BackfillPipeline, retry
from winterload.source.openmeteo import OpenMeteo, OpenMeteoResult, OpenMeteoModelResult

DayResult = dict[str, OpenMeteoResult]

class OpenMeteoBackfillPipeline(BackfillPipeline):
    @abstractmethod
    def load_day(self, client: OpenMeteo, target_day: dt.date, cities: list[Coordinate]) -> DayResult:
        """Load the open meteo data for a day, return a list of dicts per city"""

    @abstractmethod
    def transform_day(self, target_day: dt.date, openmeteo_result: OpenMeteoModelResult) -> pd.DataFrame:
        """Transform the open meteo data for a day"""

    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Returns a list of all feature columns"""

    def build_day(self, client: OpenMeteo, target_day: dt.date, cities: list[Coordinate] = COORDINATES) -> pd.DataFrame:
        city_rows: DayResult = self.load_day(client, target_day, cities)
        rows = []
        for city, openmeteo_result in city_rows.items():
            for openmeteo_model_name, openmeteo_model_result in openmeteo_result.items():
                day_hours = self.transform_day(target_day, openmeteo_model_result)
                day_hours["location_id"] = city
                day_hours["model"] = openmeteo_model_name
                rows.append(day_hours)

        result = pd.concat(rows, ignore_index=True).rename(columns={"date": "event_timestamp"})
        return result[self.feature_columns()]

    def run(self, start_date: dt.date, end_date: dt.date,
            force: bool = False, cities: list[Coordinate] = COORDINATES) -> None:
        # single-run data never changes once published -- cache it for the lifetime of a long backfill
        client = OpenMeteo(cache_expire_after=30 * 86400)
        written = skipped = failed = 0
        day = start_date
        while day <= end_date:
            self.logger.info("%s: processing %s (%s to %s)", self.name, day, start_date, end_date)
            exists = self.day_exists(day)
            if exists is None:
                failed += 1
                day += dt.timedelta(days=1)
                continue
            if not force and exists:
                skipped += 1
                day += dt.timedelta(days=1)
                continue
            if retry(lambda: self.write_day(self.build_day(client, day, cities), day),
                     f"Backfilling {day} ({len(cities)} cities)"):
                written += 1
            else:
                failed += 1
            day += dt.timedelta(days=1)

        self.logger.info(
            "Done: wrote %s new day(s), skipped %s already-backfilled day(s), "
            "rejected %s day(s) after repeated failures",
            written, skipped, failed,
        )
