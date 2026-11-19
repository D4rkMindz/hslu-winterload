"""Data drift of the model's inputs and target, measured with Evidently's DataDriftPreset and its default
per-column tests. No I/O: winterload.pipeline.monitoring.drift_report loads both windows, logs the result and
the HTML report to MLflow, and winterload.forecast.lifecycle.retrain_trigger decides what the share means.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from winterload.forecast.features.dataset import CALENDAR_COLUMNS, LAG_COLUMNS, TARGET_COLUMN, WEATHER_COLUMNS

# Everything the model sees, and the target: label drift is drift too.
DRIFT_COLUMNS = [TARGET_COLUMN, *LAG_COLUMNS, *WEATHER_COLUMNS, *CALENDAR_COLUMNS]


@dataclass(frozen=True)
class ColumnDrift:
    #: Evidently's test or distance, e.g. "K-S p_value" or "Wasserstein distance (normed)"
    method: str
    score: float
    threshold: float
    drifted: bool


@dataclass(frozen=True)
class DriftResult:
    drift_share: float
    drifted_columns: int
    columns: dict[str, ColumnDrift]
    #: The Evidently snapshot, for its HTML report and its JSON
    snapshot: Any


def _drifted(method: str, score: float, threshold: float) -> bool:
    # Evidently's tests report a p-value (drift below the threshold), its distances a distance (drift at or
    # above it).
    if "p_value" in method:
        return score < threshold
    return score >= threshold


def measure_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    """Compares `current` with `reference` on DRIFT_COLUMNS."""
    from evidently import Report  # Imported here: Evidently takes seconds to import.
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(current[DRIFT_COLUMNS], reference[DRIFT_COLUMNS])
    metrics = snapshot.dict()["metrics"]

    share, count, columns = 0.0, 0, {}
    for metric in metrics:
        config, value = metric["config"], metric["value"]
        if config["type"].endswith("DriftedColumnsCount"):
            share, count = float(value["share"]), int(value["count"])
        elif config["type"].endswith("ValueDrift"):
            method, threshold, score = config["method"], float(config["threshold"]), float(value)
            columns[config["column"]] = ColumnDrift(method, score, threshold, _drifted(method, score, threshold))
    return DriftResult(drift_share=share, drifted_columns=count, columns=columns, snapshot=snapshot)
