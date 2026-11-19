"""Hyperparameter tuning with Optuna (`-p training.tuning`, Kestra flow `tuning`). One parent run in
TUNING_EXPERIMENT, one nested run per trial.

Before the final test (test year closed), it tunes on the fixed folds at the end of TRAIN. It first scores the
preliminary model (LightGBM's defaults) on VALIDATION, which sets the target 10% below it (reviewer point
B1/A4), then scores the best trial once on VALIDATION: the validation gate that final_test checks.

After the final test, it tunes on the six newest 8-week folds before the current champion/challenger holdout,
so a monthly run follows the season. VALIDATION lies a year back by then, so these runs have no gate and only
report the folds against the seasonal-naive baseline.

Either way the result only takes effect once its tuned_settings.py artifact is committed to
winterload/forecast/model/tuned.py (config-as-code).

Not a TrainingPipeline: `-p training` (the weekly flow) must not start it.
"""

import datetime as dt
from typing import Any

import numpy as np
import optuna
import pandas as pd

from winterload.forecast.evaluation.splits import challenger_train_end, training_windows
from winterload.forecast.evaluation.target import compute_target_mape
from winterload.forecast.evaluation.tuning import (
    DEFAULT_PARAMS,
    DEFAULT_TRIALS,
    SEARCH_SEED,
    baseline_fold_mapes,
    score_folds,
    settings_from_params,
    settings_params,
    suggest_settings,
    tuned_settings_snippet,
    tuning_folds,
)
from winterload.forecast.lifecycle.final_test import BEST_PARAM_PREFIX
from winterload.forecast.model.preliminary import PreliminaryModel
from winterload.pipeline.base import Pipeline
from winterload.pipeline.training.final_test_guard import MARKER_KEY as FINAL_TEST_MARKER_KEY
from winterload.repository.champion import TUNING_EXPERIMENT, ChampionRepository
from winterload.repository.factory import champion_repository
from winterload.repository.training_frame import load_dataframe, load_train_and_validation

FIXED_FOLDS = "fixed_folds"
ROLLING_FOLDS = "rolling_folds"
# Pruning compares a trial with the median of earlier trials on the same fold, so it needs a few first.
PRUNER_STARTUP_TRIALS = 5


class TuningPipeline(Pipeline):
    name = "tuning"
    order: int = 250

    def run(self, trials: int = DEFAULT_TRIALS, reference_date: dt.date | None = None) -> None:
        reference_date = reference_date or dt.datetime.now(dt.UTC).date()
        registry = champion_repository()
        released = self.storage.read_marker(FINAL_TEST_MARKER_KEY) is not None

        # Optuna logs each trial to stderr by default, which Kestra shows as errors.
        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()

        with registry.run(self.name, TUNING_EXPERIMENT):
            run_id = registry.active_run_id()
            if released:
                history = self._rolling_history(reference_date)
                train = validation = None
                mode = ROLLING_FOLDS
                datasets = {"history": history}
            else:
                train, validation = load_train_and_validation(self.storage)
                history = train
                mode = FIXED_FOLDS
                datasets = {"train": train, "validation": validation}

            folds = tuning_folds(history)
            baseline = baseline_fold_mapes(folds)
            registry.log(
                params={"mode": mode, "trials": trials, "seed": SEARCH_SEED, "reference_date": reference_date,
                        "fold_count": len(folds), "history_rows": len(history),
                        "history_start": history["event_timestamp"].min(),
                        "history_end": history["event_timestamp"].max(),
                        "first_fold_start": folds[0][1]["event_timestamp"].min()},
                metrics={"baseline_fold_mape": float(np.mean(baseline))}
                        | {f"baseline_fold_{i}_mape": value for i, value in enumerate(baseline)},
                datasets=datasets,
            )
            self.logger.info("TUNING: %s, %s trial(s) on %s fold(s) of %s row(s)", mode, trials, len(folds),
                             len(history))

            if train is not None and validation is not None:
                preliminary_mape = PreliminaryModel().fit(train).evaluate(validation)
                target_mape = compute_target_mape(preliminary_mape)
                registry.log(params={}, metrics={"preliminary_validation_mape": preliminary_mape,
                                                 "target_mape": target_mape})
                self.logger.info("TUNING: preliminary validation MAPE %.4f%%, target %.4f%%",
                                 preliminary_mape * 100, target_mape * 100)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEARCH_SEED),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS),
            )
            study.enqueue_trial(DEFAULT_PARAMS)
            study.optimize(lambda trial: self._objective(trial, folds, registry), n_trials=trials)

            best = settings_from_params(study.best_params)
            states = [trial.state for trial in study.trials]
            registry.log(
                params={BEST_PARAM_PREFIX + name: value for name, value in settings_params(best).items()}
                       | {"best_trial": study.best_trial.number},
                metrics={"best_fold_mape": study.best_value,
                         "completed_trials": states.count(optuna.trial.TrialState.COMPLETE),
                         "pruned_trials": states.count(optuna.trial.TrialState.PRUNED)},
            )
            self.logger.info("TUNING: best trial %s, fold MAPE %.4f%% (baseline %.4f%%)", study.best_trial.number,
                             study.best_value * 100, np.mean(baseline) * 100)

            if train is not None and validation is not None:
                validation_mape = PreliminaryModel(best).fit(train).evaluate(validation)
                passed = validation_mape <= target_mape
                registry.log(params={}, metrics={"validation_mape": validation_mape})
                registry.tag({"gate_passed": passed})
                self.logger.info("TUNING: validation MAPE %.4f%%, target %.4f%%: gate %s", validation_mape * 100,
                                 target_mape * 100, "passed" if passed else "missed")

            registry.log_text(tuned_settings_snippet(best, run_id), "tuned_settings.py")
            self.logger.info("TUNING: run %s. To use these settings, paste its tuned_settings.py artifact into "
                             "src/winterload/forecast/model/tuned.py and commit it.", run_id)

    def _rolling_history(self, reference_date: dt.date) -> pd.DataFrame:
        """Everything before the current champion/challenger holdout. The folds take its last 48 weeks."""
        end = challenger_train_end(reference_date)
        frames = [load_dataframe(self.storage, start, stop) for start, stop in training_windows(end, True)]
        return pd.concat(frames, ignore_index=True)

    def _objective(self, trial: optuna.Trial, folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                   registry: ChampionRepository) -> float:
        settings = suggest_settings(trial)
        running_means: list[float] = []

        def report(fold: int, mean_so_far: float) -> None:
            running_means.append(mean_so_far)
            trial.report(mean_so_far, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        pruned = False
        with registry.run(f"trial-{trial.number:03d}", TUNING_EXPERIMENT, nested=True):
            params: dict[str, Any] = settings_params(settings) | {"trial": trial.number}
            registry.log(params=params, metrics={})
            try:
                scores = score_folds(settings, folds, report)
            except optuna.TrialPruned:
                # Caught inside the run, so MLflow ends it as FINISHED, not FAILED; re-raised below for optuna.
                pruned = True
                registry.log(params={}, metrics={"mean_fold_mape_so_far": running_means[-1]})
                registry.tag({"state": "pruned", "pruned_after_folds": len(running_means)})
            else:
                mean = float(np.mean(scores))
                registry.log(params={}, metrics={"mean_fold_mape": mean}
                             | {f"fold_{i}_mape": value for i, value in enumerate(scores)})
                registry.tag({"state": "complete"})
        if pruned:
            raise optuna.TrialPruned()
        return mean
