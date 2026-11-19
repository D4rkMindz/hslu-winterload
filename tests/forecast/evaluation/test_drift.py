import numpy as np
import pandas as pd

from winterload.forecast.evaluation.drift import DRIFT_COLUMNS, measure_drift


def _frame(rows: int, rng: np.random.Generator, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({column: rng.normal(100 + shift, 10, size=rows) for column in DRIFT_COLUMNS})


def test_a_shift_in_every_column_drifts_the_whole_frame():
    rng = np.random.default_rng(0)

    result = measure_drift(_frame(2000, rng), _frame(168, rng, shift=50))

    assert result.drift_share == 1.0
    assert result.drifted_columns == len(DRIFT_COLUMNS)
    assert set(result.columns) == set(DRIFT_COLUMNS)
    assert all(column.drifted for column in result.columns.values())


def test_identical_frames_do_not_drift():
    reference = _frame(2000, np.random.default_rng(0))

    result = measure_drift(reference, reference.copy())

    assert result.drift_share == 0.0
    assert not any(column.drifted for column in result.columns.values())


def test_only_the_monitored_columns_are_compared():
    rng = np.random.default_rng(0)
    reference, current = _frame(500, rng), _frame(168, rng)
    reference["event_timestamp"] = pd.Timestamp("2025-01-01", tz="UTC")
    current["event_timestamp"] = pd.Timestamp("2026-01-01", tz="UTC")

    assert set(measure_drift(reference, current).columns) == set(DRIFT_COLUMNS)
