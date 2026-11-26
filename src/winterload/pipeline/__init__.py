"""Runs every winterload pipeline stage once, in order: features, training, inference. The
`winterload` console script (pyproject.toml) is this module's `main()`; `-s`/`-e`/`-f` configure
the features stage's date range/force the same way `python -m winterload.pipeline.features` does
on its own -- training takes no date range, and inference takes `--target-day` and `--model` instead.
"""

import datetime as dt
import logging
import sys
from collections.abc import Callable

from winterload import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.errors import StageError
from winterload.forecast.evaluation.tuning import DEFAULT_TRIALS
from winterload.forecast.inference import CHAMPION, MODEL_ROLES
from winterload.pipeline.features import FeaturePipeline
from winterload.pipeline.features.energy_load_backfill import EnergyLoadBackfill
from winterload.pipeline.features.weather_forecast_backfill import WeatherForecastBackfill
from winterload.pipeline.features.weather_measurements_backfill import WeatherMeasurementsBackfill
from winterload.pipeline.inference import InferencePipeline
from winterload.pipeline.monitoring import MonitoringPipeline
from winterload.pipeline.training import TrainingPipeline
from winterload.pipeline.training.final_test import FinalTestPipeline
from winterload.pipeline.training.tuning import TuningPipeline

#: Run order for a plain `winterload` invocation; `-p` may instead pick any subset of these names.
#: Monitoring stays out: it runs on its own daily schedule with `-p monitoring`. So do tuning and the final
#: test, which only run on request (`-p training.tuning`, `-p training.final_test`).
DEFAULT_STAGE_ORDER = ["features", "training", "inference"]

logger = logging.getLogger("winterload.pipeline")


def run_all(steps: list[Callable[[], None]]) -> None:
    """Runs every given step once, in order. A StageError stops the run with one ERROR line and exit code 1,
    which Kestra shows as a failed execution with that line as the reason."""
    try:
        for step in steps:
            step()
    except StageError as error:
        logger.error("%s: %s", type(error).__name__, error)
        sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--start", type=dt.date.fromisoformat, default=DEFAULT_START_DATE,
        help=f"default: {DEFAULT_START_DATE.isoformat()}",
    )
    parser.add_argument(
        "-e", "--end", type=dt.date.fromisoformat, default=DEFAULT_END_DATE,
        help=f"default: {DEFAULT_END_DATE.isoformat()}",
    )
    parser.add_argument(
        "-p", "--pipeline", action="append", default=None,
        help="default: run all pipelines (feature -> training -> inference)",
    )
    parser.add_argument("-f", "--force", action="store_true", help="re-fetch days the features stage already has")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"training.tuning: Optuna trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--tuning-run-id", default=None,
                        help="training.final_test: the tuning run to check (default: TUNING_RUN_ID in "
                             "winterload/forecast/model/tuned.py)")
    parser.add_argument("--target-day", type=dt.date.fromisoformat, default=None,
                        help="inference: the Zurich local day to forecast (default: tomorrow)")
    parser.add_argument("--model", action="append", choices=MODEL_ROLES, default=None,
                        help=f"inference: one of {', '.join(MODEL_ROLES)}, repeat it for several "
                             f"(default: {CHAMPION})")
    args = parser.parse_args()

    # Each stage is bound to exactly the args it needs, right here where we know them -- the
    # features-family stages take the date range + force, inference the target day and models, training
    # none of them. That keeps every `run()` signature strict and type-checked;
    # there is no single generic call that would force them onto one shared shape.
    stages: dict[str, Callable[[], None]] = {
        "features": lambda: FeaturePipeline().run(args.start, args.end, force=args.force),
        "features.energy.backfill": lambda: EnergyLoadBackfill().run(args.start, args.end, force=args.force),
        "features.forecast.backfill": lambda: WeatherForecastBackfill().run(args.start, args.end, force=args.force),
        "features.weather.backfill": lambda: WeatherMeasurementsBackfill().run(args.start, args.end, force=args.force),
        "training": lambda: TrainingPipeline.run_all(),
        "training.tuning": lambda: TuningPipeline().run(trials=args.trials),
        # Kestra passes an empty string when its input is left blank.
        "training.final_test": lambda: FinalTestPipeline().run(tuning_run_id=args.tuning_run_id or None),
        "inference": lambda: InferencePipeline().run(target_day=args.target_day, models=args.model or [CHAMPION]),
        "monitoring": lambda: MonitoringPipeline.run_all(),
    }
    selected = args.pipeline or DEFAULT_STAGE_ORDER
    run_all([stages[name] for name in selected])
