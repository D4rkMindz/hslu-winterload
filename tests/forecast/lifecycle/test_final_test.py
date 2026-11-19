import pytest

from winterload.errors import StageError
from winterload.forecast.evaluation.tuning import settings_params
from winterload.forecast.lifecycle.final_test import (
    BEST_PARAM_PREFIX,
    FinalTestGate,
    GateNotPassed,
    SettingsMismatch,
    TrackedRun,
)
from winterload.forecast.model.preliminary import LightGBMSettings

SETTINGS = LightGBMSettings(n_estimators=731, learning_rate=0.0371, num_leaves=63, max_depth=-1,
                            min_child_samples=12, subsample=0.81, subsample_freq=1, colsample_bytree=0.66,
                            reg_lambda=0.42)


def _run(settings=SETTINGS, **metrics) -> TrackedRun:
    # MLflow hands params back as strings.
    params = {BEST_PARAM_PREFIX + name: str(value) for name, value in settings_params(settings).items()}
    return TrackedRun(run_id="abc", params=params,
                      metrics={"validation_mape": 0.050, "target_mape": 0.0512} | metrics)


def test_a_run_that_passed_with_the_committed_settings_opens_the_year():
    FinalTestGate().check(SETTINGS, _run())


def test_a_missed_gate_keeps_the_year_closed():
    with pytest.raises(GateNotPassed, match="missed the validation gate"):
        FinalTestGate().check(SETTINGS, _run(validation_mape=0.06))


def test_exactly_at_the_target_passes():
    FinalTestGate().check(SETTINGS, _run(validation_mape=0.0512))


def test_a_run_without_a_validation_gate_is_refused():
    run = TrackedRun(run_id="abc", params=_run().params, metrics={"best_fold_mape": 0.05})

    with pytest.raises(GateNotPassed, match="no validation gate"):
        FinalTestGate().check(SETTINGS, run)


def test_settings_that_differ_from_the_run_are_refused():
    other = LightGBMSettings(**settings_params(SETTINGS) | {"num_leaves": 64})

    with pytest.raises(SettingsMismatch, match="num_leaves"):
        FinalTestGate().check(other, _run())


def test_gate_errors_are_stage_errors():
    # winterload.pipeline.run_all turns StageErrors into one ERROR line and exit code 1.
    assert issubclass(GateNotPassed, StageError) and issubclass(SettingsMismatch, StageError)
