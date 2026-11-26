"""Runs the retrain triggers, reviewer point B3/A5. It scores the champion and baseline forecasts that
winterload.pipeline.inference issued on schedule over the last 7 days against the realised load, measures
the drift of the same days (drift_report), and runs champion/challenger when either trigger fires.

Only hours with a scheduled champion forecast, a scheduled baseline forecast and a realised load count.
"""

import datetime as dt

import pandas as pd

from winterload.forecast.inference import BASELINE, CHAMPION
from winterload.forecast.lifecycle.retrain_trigger import MIN_MONITORING_HOURS, RetrainTriggerPolicy, monitoring_window
from winterload.forecast.metrics import mape
from winterload.pipeline.monitoring import MonitoringPipeline
from winterload.pipeline.monitoring.drift_report import report_drift
from winterload.pipeline.training.champion_challenger import ChampionChallengerPipeline
from winterload.pipeline.training.final_test_guard import MARKER_KEY as FINAL_TEST_MARKER_KEY
from winterload.repository.champion import MONITORING_EXPERIMENT
from winterload.repository.factory import champion_repository
from winterload.repository.forecast import ForecastRepository
from winterload.repository.load import LoadRepository
from winterload.repository.training_frame import load_dataframe


class RetrainMonitorPipeline(MonitoringPipeline):
    name = "retrain_monitor"
    order: int = 410

    def run(self, reference_date: dt.date | None = None) -> None:
        reference_date = reference_date or dt.datetime.now(dt.UTC).date()

        window = monitoring_window(reference_date)
        if window is None:
            self.logger.info("RETRAIN MONITOR: skipped, the window before %s reaches into the final test year.",
                             reference_date)
            return
        start, end = window

        registry = champion_repository()
        forecasts = ForecastRepository(self.storage)
        scored = pd.concat({"actual": LoadRepository(self.storage).hourly(start, end),
                            "champion": forecasts.scheduled(start, end, CHAMPION),
                            "baseline": forecasts.scheduled(start, end, BASELINE)}, axis=1).dropna()
        if len(scored) < MIN_MONITORING_HOURS:
            self.logger.warning("RETRAIN MONITOR: skipped, %s..%s has %s hour(s) of scheduled forecasts with "
                                "realised load, below the minimum of %s.",
                                start, end, len(scored), MIN_MONITORING_HOURS)
            return

        champion = registry.load_champion()
        if champion is None:
            self.logger.info("RETRAIN MONITOR: skipped, no run has promoted a champion yet.")
            return

        released = self.storage.read_marker(FINAL_TEST_MARKER_KEY) is not None
        frame = load_dataframe(self.storage, start, end)
        drift = (report_drift(self.storage, registry, champion, frame, window, released, reference_date, self.logger)
                 if len(frame) >= MIN_MONITORING_HOURS else None)

        decision = RetrainTriggerPolicy().decide(
            mape(scored["actual"].to_numpy(), scored["champion"].to_numpy()),
            mape(scored["actual"].to_numpy(), scored["baseline"].to_numpy()),
        )
        drift_retrain = drift is not None and drift.retrain
        retrain = decision.retrain or drift_retrain
        reasons = [reason for reason, fired in (("mape", decision.retrain), ("drift", drift_retrain)) if fired]
        self.logger.info(
            "RETRAIN MONITOR: %s..%s (%s row(s)), model MAPE=%.4f%%, baseline MAPE=%.4f%%, threshold=%.4f%%, "
            "drift share=%s: %s",
            start, end, len(scored), decision.model_mape * 100, decision.baseline_mape * 100,
            decision.threshold_mape * 100, f"{drift.drift_share:.2f}" if drift else "n/a",
            f"retrain ({', '.join(reasons)})" if retrain else "keep",
        )
        metrics = {"model_mape_7d": decision.model_mape, "baseline_mape_7d": decision.baseline_mape,
                   "threshold_mape_7d": decision.threshold_mape}
        if drift is not None:
            metrics |= {"drift_share": drift.drift_share, "drift_threshold_share": drift.threshold_share}
        with registry.run(self.name, MONITORING_EXPERIMENT):
            registry.log(
                params={"reference_date": reference_date, "rows": len(scored), "retrain": retrain,
                        "mape_trigger": decision.retrain, "drift_trigger": drift_retrain,
                        "retrain_reason": "+".join(reasons) or "none"},
                metrics=metrics,
            )

        if retrain:
            ChampionChallengerPipeline(logger=self.logger, storage=self.storage).run(reference_date)
