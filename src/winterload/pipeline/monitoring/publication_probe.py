"""Logs how fresh the inputs of the 08:00 forecast are, reviewer point A8. Each run appends one
observation per source to the result store, so repeated runs collect the delays and revisions that
the 48h lag rule and the weather run choice depend on.

This is the one step that has to ask the sources themselves, because it measures them. Notebooks read
what it logged. Run it on its own with `python -m winterload.pipeline.monitoring.publication_probe`.
"""

import dataclasses
import datetime as dt
import logging

import pandas as pd
import requests
from winterload.forecast.evaluation.backtest import compare_load_versions, observe_publication, observe_weather_run
from winterload.pipeline.monitoring import MonitoringPipeline
from winterload.repository.factory import result_repository
from winterload.repository.load import LoadRepository
from winterload.repository.results import LOAD_SNAPSHOTS, PUBLICATION_LOG, REVISION_LOG, WEATHER_RUN_LOG, \
    ResultRepository
from winterload.source.entso_e import EntsoE
from winterload.source.frauenhofer_ise import FrauenhoferISE
from winterload.source.openmeteo import OpenMeteo
from winterload.storage import Storage

LOOKBACK = pd.Timedelta(3, unit="days")
# One week before the test year, checked against a fresh download for late revisions in older data.
REVISION_CHECK_START = dt.date(2025, 7, 1)
REVISION_CHECK_END = dt.date(2025, 7, 8)
SOURCE_ERRORS = (requests.RequestException, KeyError, ValueError)


def _entsoe_load(start: dt.datetime, end: dt.datetime) -> pd.Series:
    rows = EntsoE().get_load(start_date=start, end_date=end)
    series = pd.DataFrame(rows).set_index("timestamp")["quantity"]
    return series[~series.index.duplicated()]


def _energy_charts_load(start: dt.datetime, end: dt.datetime) -> pd.Series:
    rows = FrauenhoferISE().get_public_power(start=start, end=end)
    series = pd.Series({pd.Timestamp(r["timestamp"]).tz_convert("UTC"): r["load"] for r in rows if r["load"] is not None},
                       dtype="float64")
    return series[~series.index.duplicated()]


class PublicationProbe(MonitoringPipeline):
    name = "publication_probe"
    order: int = 420

    def __init__(self, logger: logging.Logger | None = None, storage: Storage | None = None,
                 results: ResultRepository | None = None) -> None:
        super().__init__(logger, storage)
        self.results = results or result_repository()

    def run(self, now: pd.Timestamp | None = None) -> None:
        now = now if now is not None else pd.Timestamp.now(tz="UTC").floor("s")
        loads = self._load_sources(now)
        self._append(PUBLICATION_LOG, pd.DataFrame([
            dataclasses.asdict(observe_publication(name, now, series.index.max())) for name, series in loads.items()
        ]), now)
        self._log_revisions(loads, now)
        self._log_stored_revision(now)
        self._log_weather_runs(now)

    def _append(self, name: str, rows: pd.DataFrame, now: pd.Timestamp) -> None:
        self.results.append(name, rows.assign(run_at=now))

    def _load_sources(self, now: pd.Timestamp) -> dict[str, pd.Series]:
        """A source that fails gets logged and left out, so the other one still counts."""
        start = (now - LOOKBACK).to_pydatetime()
        end = now.to_pydatetime()
        naive_start = (now - LOOKBACK).tz_localize(None).to_pydatetime()
        naive_end = now.tz_localize(None).to_pydatetime()
        loads = {}
        for name, fetch in [("entsoe", lambda: _entsoe_load(start, end)),
                            ("energy_charts", lambda: _energy_charts_load(naive_start, naive_end))]:
            try:
                loads[name] = fetch()
            except SOURCE_ERRORS as error:
                self.logger.warning("PUBLICATION PROBE: %s unavailable: %s", name, error)
        return loads

    def _log_revisions(self, loads: dict[str, pd.Series], now: pd.Timestamp) -> None:
        """Compares each source against the value first seen for each hour."""
        previous = self.results.read(LOAD_SNAPSHOTS, parse_dates=["event_timestamp"])
        revisions = []
        for name, series in loads.items():
            if not previous.empty:
                first_seen = (previous.loc[previous["source"] == name]
                              .drop_duplicates("event_timestamp").set_index("event_timestamp")["quantity"])
                revisions.append({"source": name, "window": "earlier snapshot",
                                  **dataclasses.asdict(compare_load_versions(first_seen, series))})
            self._append(LOAD_SNAPSHOTS, series.rename("quantity").rename_axis("event_timestamp").reset_index().assign(source=name), now)
        if revisions:
            self._append(REVISION_LOG, pd.DataFrame(revisions), now)

    def _log_stored_revision(self, now: pd.Timestamp) -> None:
        stored = LoadRepository(self.storage).hourly(REVISION_CHECK_START, REVISION_CHECK_END)
        try:
            fresh = _entsoe_load(dt.datetime.combine(REVISION_CHECK_START, dt.time.min),
                                 dt.datetime.combine(REVISION_CHECK_END, dt.time.min))
        except SOURCE_ERRORS as error:
            self.logger.warning("PUBLICATION PROBE: entsoe unavailable for the stored week: %s", error)
            return
        window = f"stored {REVISION_CHECK_START} to {REVISION_CHECK_END - dt.timedelta(days=1)}"
        self._append(REVISION_LOG, pd.DataFrame([{"source": "entsoe", "window": window,
                                                  **dataclasses.asdict(compare_load_versions(stored, fresh))}]), now)

    def _log_weather_runs(self, now: pd.Timestamp) -> None:
        observations = []
        for host_url in (OpenMeteo.SINGLE_RUNS_URL, OpenMeteo.FORECAST_URL):
            metadata = OpenMeteo.get_run_metadata(host_url)
            observations.append(dataclasses.asdict(observe_weather_run(
                host_url,
                now,
                pd.Timestamp(metadata["last_run_initialisation_time"], unit="s", tz="UTC"),
                pd.Timestamp(metadata["last_run_availability_time"], unit="s", tz="UTC"),
            )))
        self._append(WEATHER_RUN_LOG, pd.DataFrame(observations), now)


if __name__ == "__main__":
    PublicationProbe().run()
