"""The one-time final test on the held-out year (`-p training.final_test`, Kestra flow `final_test`, manual
only), reviewer point B1/A4 and docs/proposal.pdf, section 1.

It first checks, without touching the year, that the committed TUNED_SETTINGS passed the validation gate:
their tuning run must exist in MLflow, hold a validation MAPE at or below its target, and hold exactly these
settings (winterload.forecast.lifecycle.final_test). Any failure stops the stage with one ERROR line and
exit code 1, and the year stays closed. Then the tuned model fits on TRAIN and VALIDATION and scores the test
year against the seasonal-naive baseline. The model is logged, not registered: live serving goes through
champion/challenger.

Not a TrainingPipeline: `-p training` (the weekly flow) must not start it.
"""

import pandas as pd

from winterload.forecast.evaluation.splits import TEST_END, TEST_START, TRAIN_START, VALIDATION_END
from winterload.forecast.evaluation.tuning import settings_params
from winterload.forecast.features.dataset import TARGET_COLUMN
from winterload.forecast.lifecycle.final_test import FINAL_TARGET_MAPE, FinalTestGate, NoTunedSettings
from winterload.forecast.metrics import mae, mape
from winterload.forecast.model.baseline import seasonal_naive_forecast
from winterload.forecast.model.preliminary import PreliminaryModel
from winterload.forecast.model.tuned import TUNED_SETTINGS, TUNING_RUN_ID
from winterload.pipeline.base import Pipeline
from winterload.pipeline.training.final_test_guard import load_final_test_year
from winterload.repository.champion import TRAINING_EXPERIMENT
from winterload.repository.factory import champion_repository
from winterload.repository.training_frame import load_dataframe


class FinalTestPipeline(Pipeline):
    name = "final_test"
    order: int = 260

    def run(self, tuning_run_id: str | None = None) -> None:
        if TUNED_SETTINGS is None or TUNING_RUN_ID is None:
            raise NoTunedSettings(
                "No TUNED_SETTINGS committed in src/winterload/forecast/model/tuned.py. Run the `tuning` flow, "
                "paste its tuned_settings.py artifact there and commit it before the final test."
            )
        run_id = tuning_run_id or TUNING_RUN_ID
        registry = champion_repository()
        FinalTestGate().check(TUNED_SETTINGS, registry.get_run(run_id))
        self.logger.info("FINAL TEST: tuning run %s passed the validation gate with the committed settings", run_id)

        train = load_dataframe(self.storage, TRAIN_START, VALIDATION_END)
        model = PreliminaryModel(TUNED_SETTINGS).fit(train)

        # The run starts before the year opens, so a failure after the access marker still leaves a trace.
        with registry.run(self.name, TRAINING_EXPERIMENT):
            registry.log(params={"tuning_run_id": run_id, "train_rows": len(train), "test_start": TEST_START,
                                 "test_end": TEST_END} | settings_params(TUNED_SETTINGS), metrics={})
            test = load_final_test_year(self.storage)

            actual = test[TARGET_COLUMN].to_numpy()
            forecast = model.predict(test)
            baseline = seasonal_naive_forecast(test)
            test_mape = mape(actual, forecast)
            passed = test_mape <= FINAL_TARGET_MAPE
            registry.log(
                params={"test_rows": len(test)},
                metrics={"test_mape": test_mape, "test_mae": mae(actual, forecast),
                         "baseline_test_mape": mape(actual, baseline), "baseline_test_mae": mae(actual, baseline),
                         "target_test_mape": FINAL_TARGET_MAPE},
                datasets={"train": train, "test": test},
            )
            registry.tag({"target_met": passed})
            registry.log_dict(self._by_month(test, forecast, baseline), "test_mape_by_month.json")
            registry.log_unregistered_model(model, "final_test_model")

        self.logger.info("FINAL TEST: MAPE %.4f%% (baseline %.4f%%, target %.4f%%): target %s", test_mape * 100,
                         mape(actual, baseline) * 100, FINAL_TARGET_MAPE * 100, "met" if passed else "missed")

    @staticmethod
    def _by_month(test: pd.DataFrame, forecast, baseline) -> dict[str, dict[str, float]]:
        frame = pd.DataFrame({"month": pd.to_datetime(test["event_timestamp"]).dt.strftime("%Y-%m"),
                              "actual": test[TARGET_COLUMN].to_numpy(), "forecast": forecast, "baseline": baseline})
        return {month: {"model_mape": mape(rows["actual"].to_numpy(), rows["forecast"].to_numpy()),
                        "baseline_mape": mape(rows["actual"].to_numpy(), rows["baseline"].to_numpy()),
                        "hours": len(rows)}
                for month, rows in frame.groupby("month")}
