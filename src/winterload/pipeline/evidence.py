"""Exports the evidence of the scheduled runs to docs/runs: Kestra executions and logs, what the runs left in the
stores, and the MLflow runs. It is no `Pipeline`, so no `run_all` picks it up; `-p evidence` runs it."""

import datetime as dt
import logging

import pandas as pd

from winterload import COORDINATES
from winterload.forecast.evidence import execution_rows, log_text, missing_runs
from winterload.forecast.features.completeness import local_hours
from winterload.forecast.features.dataset import ZURICH_TZ
from winterload.forecast.inference import MODEL_ROLES
from winterload.repository.champion import MONITORING_EXPERIMENT, TRAINING_EXPERIMENT
from winterload.repository.factory import champion_repository, evidence_repository
from winterload.repository.forecast import ForecastRepository
from winterload.repository.load import LoadRepository
from winterload.repository.weather import WeatherRepository
from winterload.source.kestra import KestraExport
from winterload.storage import Storage, get_storage


class EvidenceExport:
    def __init__(self, logger: logging.Logger | None = None, storage: Storage | None = None) -> None:
        self.storage = storage or get_storage()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def run(self, export: KestraExport, first_day: dt.date) -> None:
        results = evidence_repository()
        today = pd.Timestamp.now(tz=ZURICH_TZ).date()
        rows = execution_rows(export.executions())
        results.write("executions", rows)
        # Expected runs start with the first scheduled one; days before it were setup.
        scheduled = rows.loc[rows["scheduled_date"] != "", "scheduled_date"]
        first_scheduled = dt.date.fromisoformat(scheduled.min()) if len(scheduled) else first_day
        results.write("missing_runs", missing_runs(rows, first_scheduled, today))
        lines_of: dict[str, list[dict]] = {}
        for line in export.logs():
            lines_of.setdefault(line["executionId"], []).append(line)
        for row in rows.itertuples():
            name = f"logs/{row.flow}/{row.scheduled_date or _zurich_date(row.started_at)}_{row.execution_id}"
            results.write_text(name, log_text(lines_of.get(row.execution_id, [])))
        results.write("outputs", self._outputs(first_day, today))
        since = dt.datetime.combine(first_day, dt.time(), tzinfo=dt.UTC)
        champion = champion_repository()
        runs = [champion.runs_since(experiment, since) for experiment in (TRAINING_EXPERIMENT, MONITORING_EXPERIMENT)]
        results.write("training_runs", pd.concat(runs, ignore_index=True))
        self.logger.info("EVIDENCE: wrote %s execution(s) to docs/runs", len(rows))

    def _outputs(self, first_day: dt.date, today: dt.date) -> pd.DataFrame:
        """One row per Zurich day from `first_day` to yesterday: what the runs left in the stores."""
        days = [day.date() for day in pd.date_range(first_day, today - dt.timedelta(days=1))]
        if not days:
            return pd.DataFrame()
        # A Zurich day starts at 23:00 UTC the day before, so read one day early.
        start, end = days[0] - dt.timedelta(days=1), today
        load = LoadRepository(self.storage).hourly(start, end).dropna()
        weather = WeatherRepository(self.storage).forecast(start, end)
        weather_days = pd.to_datetime(weather["event_timestamp"], utc=True).dt.tz_convert(ZURICH_TZ).dt.date
        weather_counts = weather_days.value_counts()
        forecasts = {role: ForecastRepository(self.storage).scheduled(start, end, role) for role in MODEL_ROLES}
        rows = []
        for day in days:
            row = {"date": day.isoformat(),
                   # Energy days are UTC days.
                   "energy_load_hours": int((load.index.date == day).sum()),
                   "weather_forecast_rows": int(weather_counts.get(day, 0))}
            for role, series in forecasts.items():
                row[f"forecasts_{role}"] = int((series.index.tz_convert(ZURICH_TZ).date == day).sum())
            row["complete"] = (row["energy_load_hours"] == 24
                               and row["weather_forecast_rows"] == len(COORDINATES) * local_hours(day))
            rows.append(row)
        return pd.DataFrame(rows)


def _zurich_date(timestamp: str) -> dt.date:
    return pd.Timestamp(timestamp).tz_convert(ZURICH_TZ).date()
