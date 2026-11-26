"""Forecasts every hour of a Zurich local day with the models asked for, and appends the forecasts to
winterload.repository.forecast. A rerun only adds rows: the newest run wins in the latest view, and every row
keeps the model version that made it.

A model that fails to load, or an hour whose inputs are missing (at 07:00 the 48h load lag of tomorrow's evening
hours is often not published yet), doesn't stop the others. The stage then ends with InferenceIncomplete: Kestra
marks the execution failed and mails, and a later rerun fills the gaps.
"""

import datetime as dt
import uuid
from collections.abc import Sequence

import numpy as np

from winterload.config import LOG_FORMAT
from winterload.errors import StageError
from winterload.forecast.inference import (
    BASELINE,
    BASELINE_MODEL_ID,
    CHAMPION,
    ForecastRun,
    ModelForecast,
    split_complete,
    tomorrow_in_zurich,
)
from winterload.forecast.model.baseline import BASELINE_COLUMN, seasonal_naive_forecast
from winterload.log_format import kestra_outputs
from winterload.pipeline.base import Pipeline
from winterload.repository.factory import champion_repository
from winterload.repository.forecast import ForecastRepository
from winterload.repository.training_frame import load_inference_frame


class InferenceIncomplete(StageError):
    """A model failed or skipped hours. The forecasts the run could make are stored."""


class InferencePipeline(Pipeline):
    name = "inference"
    order: int = 300

    def run(self, target_day: dt.date | None = None, models: Sequence[str] = (CHAMPION,)) -> None:
        target_day = target_day or tomorrow_in_zurich(dt.datetime.now(dt.UTC))
        started_at = dt.datetime.now(dt.UTC)
        run_id = str(uuid.uuid4())
        frame = load_inference_frame(self.storage, target_day)
        forecasts: list[ModelForecast] = []
        problems: list[str] = []
        registry = None
        for role in dict.fromkeys(models):
            # KeyError: the model wants a feature column the frame lacks. It fails this model, not the run.
            try:
                if role == BASELINE:
                    model_id, columns, predict = BASELINE_MODEL_ID, [BASELINE_COLUMN], seasonal_naive_forecast
                else:
                    # Only built when a registry model is asked for, so a baseline run needs no MLflow.
                    registry = registry or champion_repository()
                    model, model_id = registry.load_model(role)
                    columns, predict = model.feature_columns, model.predict
                complete, missing = split_complete(frame, columns)
                result = np.asarray(predict(complete), dtype="float64") if len(complete) else np.array([])
            except (StageError, KeyError) as error:
                self.logger.error("INFERENCE: %s failed: %s: %s", role, type(error).__name__, error)
                problems.append(f"{role} failed ({type(error).__name__})")
                continue
            if missing:
                self.logger.warning("INFERENCE: %s skipped %s hour(s) with missing inputs: %s", role, len(missing),
                                    ", ".join(time.isoformat() for time in missing))
                problems.append(f"{role} skipped {len(missing)} hour(s)")
            if len(result):
                forecasts.append(ModelForecast(model_id, role, complete["event_timestamp"], result))
        run = ForecastRun(run_id, started_at, dt.datetime.now(dt.UTC), target_day)
        written = ForecastRepository(self.storage).save(run, forecasts)
        self.logger.info("INFERENCE: run %s for %s (%s hour(s)) wrote %s forecast row(s) for %s", run_id,
                         target_day, len(frame), written,
                         ", ".join(forecast.model_role for forecast in forecasts) or "no model")
        if problems:
            summary = f"{target_day}: " + "; ".join(problems)
            if LOG_FORMAT == "kestra":
                print(kestra_outputs({"problems": summary}), flush=True)
            raise InferenceIncomplete(summary)


def main() -> None:
    InferencePipeline().run()
