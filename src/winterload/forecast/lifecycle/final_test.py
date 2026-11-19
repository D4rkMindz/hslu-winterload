"""Decides whether the final test may open the held-out year, reviewer point B1/A4. The year may be read
once, so the settings it tests must have passed the validation gate first: the tuning run committed next to
TUNED_SETTINGS (winterload.forecast.model.tuned) must hold a validation MAPE at or below its target, and its
best settings must be the committed ones. winterload.pipeline.training.final_test runs this check against
MLflow before it touches the year.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from winterload.errors import StageError
from winterload.forecast.evaluation.tuning import settings_params
from winterload.forecast.model.preliminary import LightGBMSettings

# docs/proposal.pdf, section 1: MAPE <= 6.5% on the held-out year, at least 15% below the baseline's 7.6%.
FINAL_TARGET_MAPE = 0.065
# Tuning logs its best settings as `best_<name>` params.
BEST_PARAM_PREFIX = "best_"


class NoTunedSettings(StageError):
    """Nothing is committed in winterload.forecast.model.tuned yet."""


class GateNotPassed(StageError):
    """The tuning run has no validation gate, or missed it."""


class SettingsMismatch(StageError):
    """The tuning run's best settings differ from the committed TUNED_SETTINGS."""


@dataclass(frozen=True)
class TrackedRun:
    """What the gate reads from an MLflow run. MLflow keeps params as strings."""
    run_id: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


def _same(expected: Any, logged: str | None) -> bool:
    if logged is None:
        return False
    if expected is None:
        return logged == "None"
    if isinstance(expected, bool | int):
        return str(expected) == logged
    return math.isclose(float(expected), float(logged), rel_tol=1e-9)


class FinalTestGate:
    def check(self, settings: LightGBMSettings, run: TrackedRun) -> None:
        """Raises a StageError unless `run` passed the validation gate with exactly `settings`."""
        mismatches = [
            f"{name}: committed {value!r}, run {run.params.get(BEST_PARAM_PREFIX + name)!r}"
            for name, value in settings_params(settings).items()
            if not _same(value, run.params.get(BEST_PARAM_PREFIX + name))
        ]
        if mismatches:
            raise SettingsMismatch(
                f"Tuning run {run.run_id} does not hold the committed TUNED_SETTINGS ({'; '.join(mismatches)}). "
                "Paste that run's tuned_settings.py artifact into winterload/forecast/model/tuned.py, or pass the "
                "run that produced the committed settings."
            )

        validation_mape = run.metrics.get("validation_mape")
        target_mape = run.metrics.get("target_mape")
        if validation_mape is None or target_mape is None:
            raise GateNotPassed(
                f"Tuning run {run.run_id} has no validation gate (validation_mape/target_mape). Only a tuning run "
                "before the final test, on the fixed folds, scores VALIDATION."
            )
        if validation_mape > target_mape:
            raise GateNotPassed(
                f"Tuning run {run.run_id} missed the validation gate: MAPE {validation_mape:.4%} > target "
                f"{target_mape:.4%}. The test year stays closed."
            )
