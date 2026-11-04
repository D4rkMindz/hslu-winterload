"""Runs the retrain trigger, reviewer point B3/A5. It scores `champion_eval` and the seasonal-naive
baseline on the last 7 days and runs champion/challenger when the trigger fires.

`champion_eval` stands in for the serving model. The serving model refits on all data, so it has seen
some of these hours, and winterload.pipeline.inference does not log real forecasts yet.
`champion_eval` is older, which makes the trigger fire earlier, never later.
"""

import datetime as dt

from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.lifecycle.retrain_trigger import MIN_MONITORING_HOURS, RetrainTriggerPolicy, monitoring_window
from winterload.forecast.metrics import mape
from winterload.forecast.model.baseline import seasonal_naive_forecast
from winterload.pipeline.monitoring import MonitoringPipeline
from winterload.pipeline.training.champion_challenger import ChampionChallengerPipeline
from winterload.repository.factory import champion_repository
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

        frame = load_dataframe(self.storage, start, end)
        if len(frame) < MIN_MONITORING_HOURS:
            self.logger.warning("RETRAIN MONITOR: skipped, %s..%s has %s row(s), below the minimum of %s.",
                                start, end, len(frame), MIN_MONITORING_HOURS)
            return

        registry = champion_repository()
        champion = registry.load_champion()
        if champion is None:
            self.logger.info("RETRAIN MONITOR: skipped, no run has promoted a champion yet.")
            return

        decision = RetrainTriggerPolicy().decide(
            champion.evaluation_model.evaluate(frame),
            mape(frame[TARGET_COLUMN].to_numpy(), seasonal_naive_forecast(frame)),
        )
        self.logger.info(
            "RETRAIN MONITOR: %s..%s (%s row(s)), model MAPE=%.4f%%, baseline MAPE=%.4f%%, threshold=%.4f%%: %s",
            start, end, len(frame), decision.model_mape * 100, decision.baseline_mape * 100,
            decision.threshold_mape * 100, "retrain" if decision.retrain else "keep",
        )
        with registry.run(self.name):
            registry.log(
                params={"reference_date": reference_date, "rows": len(frame), "retrain": decision.retrain},
                metrics={"model_mape_7d": decision.model_mape, "baseline_mape_7d": decision.baseline_mape,
                         "threshold_mape_7d": decision.threshold_mape},
            )

        if decision.retrain:
            ChampionChallengerPipeline(logger=self.logger, storage=self.storage).run(reference_date)
