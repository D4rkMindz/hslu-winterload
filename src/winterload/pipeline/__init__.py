"""Runs every winterload pipeline stage once, in order: features, training, inference. The
`winterload` console script (pyproject.toml) is this module's `main()`; `-s`/`-e`/`-f` configure
the features stage's date range/force the same way `python -m winterload.pipeline.features` does
on its own -- training takes no date range, and inference takes `--target-day` and `--model` instead.
"""

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from winterload import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.config import settings
from winterload.errors import StageError
from winterload.forecast.evaluation.tuning import DEFAULT_TRIALS
from winterload.forecast.inference import CHAMPION, MODEL_ROLES
from winterload.pipeline.evidence import EvidenceExport
from winterload.pipeline.features import FeaturePipeline
from winterload.pipeline.features.energy_load_backfill import EnergyLoadBackfill
from winterload.pipeline.features.weather_forecast_backfill import WeatherForecastBackfill
from winterload.pipeline.features.weather_measurements_backfill import WeatherMeasurementsBackfill
from winterload.pipeline.inference import InferencePipeline
from winterload.pipeline.monitoring import MonitoringPipeline
from winterload.pipeline.training import TrainingPipeline
from winterload.pipeline.training.final_test import FinalTestPipeline
from winterload.pipeline.training.tuning import TuningPipeline
from winterload.source.kestra import KestraExport

#: Run order for a plain `winterload` invocation; `-p` may instead pick any subset of these names.
#: Monitoring stays out: it runs on its own daily schedule with `-p monitoring`. So do tuning and the final
#: test, which only run on request (`-p training.tuning`, `-p training.final_test`).
DEFAULT_STAGE_ORDER = ["features", "training", "inference"]

STAGE_NAMES = [
    "features",
    "features.energy.backfill",
    "features.forecast.backfill",
    "features.weather.backfill",
    "training",
    "training.tuning",
    "training.final_test",
    "inference",
    "monitoring",
    "evidence",
]

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


def build_parser() -> argparse.ArgumentParser:
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
        "-p", "--pipeline", action="append", choices=STAGE_NAMES, default=None,
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
    parser.add_argument("--scheduled-at", default="",
                        help="inference: the Kestra schedule date; set means a scheduled run")
    parser.add_argument("--kestra-execution-id", default="")
    parser.add_argument("--evidence-from", default="",
                        help="evidence: folder with executions.json and logs*.json from atc "
                             "(default: fetch from the local Kestra)")
    parser.add_argument("--since", type=dt.date.fromisoformat, default=dt.date(2026, 11, 6),
                        help="evidence: first day to cover (default: 2026-11-06)")
    return parser


def evidence_export(args: argparse.Namespace) -> KestraExport:
    if args.evidence_from:
        return KestraExport.from_files(Path(args.evidence_from))
    since = dt.datetime.combine(args.since, dt.time(), tzinfo=dt.UTC)
    return KestraExport.from_api(f"http://localhost:{settings.kestra_port}", settings.kestra_ui_user,
                                 settings.kestra_ui_password, since)


def main() -> None:
    args = build_parser().parse_args()

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
        "inference": lambda: InferencePipeline().run(
            target_day=args.target_day, models=args.model or [CHAMPION], scheduled_at=args.scheduled_at,
            kestra_execution_id=args.kestra_execution_id),
        "monitoring": lambda: MonitoringPipeline.run_all(),
        "evidence": lambda: EvidenceExport().run(evidence_export(args), args.since),
    }
    # -p only accepts STAGE_NAMES, so a stage missing there could never be selected.
    assert list(stages) == STAGE_NAMES, "STAGE_NAMES is out of sync with the stages above"
    selected = args.pipeline or DEFAULT_STAGE_ORDER
    run_all([stages[name] for name in selected])
