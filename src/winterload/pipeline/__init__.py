"""Runs every winterload pipeline stage once, in order: features, training, inference. The
`winterload` console script (pyproject.toml) is this module's `main()`; `-s`/`-e`/`-f` configure
the features stage's date range/force the same way `python -m winterload.pipeline.features` does
on its own -- training and inference don't take a date range (yet), so they're always run with
their own defaults.
"""

import datetime as dt
from typing import Callable

from winterload import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.pipeline.features import FeaturePipeline
from winterload.pipeline.features.energy_load_backfill import EnergyLoadBackfill
from winterload.pipeline.features.weather_forecast_backfill import WeatherForecastBackfill
from winterload.pipeline.features.weather_measurements_backfill import WeatherMeasurementsBackfill
from winterload.pipeline.inference import InferencePipeline
from winterload.pipeline.training import TrainingPipeline

#: Run order for a plain `winterload` invocation; `-p` may instead pick any subset of these names.
DEFAULT_STAGE_ORDER = ["features", "training", "inference"]


def run_all(steps: list[Callable[[], None]]) -> None:
    """Runs every given step once, in order."""
    for step in steps:
        step()


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
    args = parser.parse_args()

    # Each stage is bound to exactly the args it needs, right here where we know them -- the
    # features-family stages take the date range + force, training/inference take neither (they
    # don't have a date range yet). That keeps every `run()` signature strict and type-checked;
    # there is no single generic call that would force them onto one shared shape.
    stages: dict[str, Callable[[], None]] = {
        "features": lambda: FeaturePipeline().run(args.start, args.end, force=args.force),
        "features.energy.backfill": lambda: EnergyLoadBackfill().run(args.start, args.end, force=args.force),
        "features.forecast.backfill": lambda: WeatherForecastBackfill().run(args.start, args.end, force=args.force),
        "features.weather.backfill": lambda: WeatherMeasurementsBackfill().run(args.start, args.end, force=args.force),
        "training": lambda: TrainingPipeline.run_all(),
        "inference": lambda: InferencePipeline.run_all(),
    }
    selected = args.pipeline or DEFAULT_STAGE_ORDER
    run_all([stages[name] for name in selected])
