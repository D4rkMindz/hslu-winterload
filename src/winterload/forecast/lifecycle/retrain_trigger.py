"""Decides whether the model has lost its edge over the seasonal-naive baseline, reviewer point
B3/A5. Baseline MAPE moves between 4.6% and 10.8% by month (docs/proposal.pdf), so the proposal's
flat 8% threshold means something different in each season. A threshold relative to the baseline
moves with it.

Evidently drift is a second trigger, as in the proposal (the MS2 summary had dropped it): a retrain also
starts when at least half of the monitored columns drifted. Either trigger suffices. champion/challenger
still only promotes a challenger that beats the champion, so a drift retrain that doesn't help changes
nothing. winterload.pipeline.monitoring.retrain_monitor runs both policies.
"""

import datetime as dt
from dataclasses import dataclass

from winterload.forecast.evaluation.splits import TEST_END, training_windows

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


# Evidently's own default: the dataset drifted when half of its columns did.
DRIFT_SHARE_THRESHOLD = 0.5
# 52 weeks, so the reference starts on the same weekday as the monitored window.
YEAR_EARLIER = dt.timedelta(weeks=52)
CHAMPION_TRAINING_FRAME = "champion_training_frame"
SAME_WEEKS_LAST_YEAR = "same_weeks_last_year"


@dataclass(frozen=True)
class DriftDecision:
    retrain: bool
    drift_share: float
    threshold_share: float


class DriftTriggerPolicy:
    def decide(self, drift_share: float) -> DriftDecision:
        """`drift_share` is the share of monitored columns whose drift test fired (Evidently)."""
        return DriftDecision(retrain=drift_share >= DRIFT_SHARE_THRESHOLD, drift_share=drift_share,
                             threshold_share=DRIFT_SHARE_THRESHOLD)


@dataclass(frozen=True)
class DriftReference:
    kind: str
    windows: list[tuple[dt.date, dt.date]]


def drift_reference(start: dt.date, end: dt.date, test_year_released: bool,
                    champion_training_end: dt.date | None) -> DriftReference | None:
    """The `[start, end)` windows the monitored window `[start, end)` is compared against.

    The same weeks one year earlier fall into the final test year until autumn 2027, and that year stays
    closed until the final test (reviewer point B1/A4). Until then the reference is what the champion
    trained on: did the inputs leave what the model learned from? None if there is no champion yet.
    """
    if test_year_released:
        return DriftReference(SAME_WEEKS_LAST_YEAR, [(start - YEAR_EARLIER, end - YEAR_EARLIER)])
    if champion_training_end is None:
        return None
    return DriftReference(CHAMPION_TRAINING_FRAME, training_windows(champion_training_end, test_year_released=False))
