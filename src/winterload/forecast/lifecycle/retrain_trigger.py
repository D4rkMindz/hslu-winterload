"""Decides whether the model has lost its edge over the seasonal-naive baseline, reviewer point
B3/A5. Baseline MAPE moves between 4.6% and 10.8% by month (docs/proposal.pdf), so the proposal's
flat 8% threshold means something different in each season. A threshold relative to the baseline
moves with it.

The proposal also retrained on Evidently drift. Evidently would flag the October to January
temperature shift, which the seasonal features already explain, so drift only gets reported from now
on and never triggers a retrain. winterload.pipeline.monitoring.retrain_monitor runs this policy.
"""

import datetime as dt
from dataclasses import dataclass

from winterload.forecast.evaluation.splits import TEST_END

RETRAIN_BASELINE_RATIO = 0.9
ROLLING_WINDOW_DAYS = 7
# Both MAPEs score the same rows, so a gap stays fair. Below 5 days the result says little.
MIN_MONITORING_HOURS = 5 * 24


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    model_mape: float
    baseline_mape: float
    threshold_mape: float


class RetrainTriggerPolicy:
    def decide(self, model_mape: float, baseline_mape: float) -> RetrainDecision:
        """Both MAPEs must cover the same hours of the last `ROLLING_WINDOW_DAYS` days."""
        threshold_mape = RETRAIN_BASELINE_RATIO * baseline_mape
        return RetrainDecision(
            retrain=model_mape > threshold_mape,
            model_mape=model_mape,
            baseline_mape=baseline_mape,
            threshold_mape=threshold_mape,
        )


def monitoring_window(reference_date: dt.date) -> tuple[dt.date, dt.date] | None:
    """The `[start, end)` window the trigger scores. Returns None while it would reach back into the
    final test year, which stays unscored until the final test (reviewer point B1/A4)."""
    start = reference_date - dt.timedelta(days=ROLLING_WINDOW_DAYS)
    if start < TEST_END:
        return None
    return start, reference_date
