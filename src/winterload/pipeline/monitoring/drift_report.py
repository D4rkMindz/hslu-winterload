"""The drift step of daily monitoring: Evidently compares the last 7 days with a reference
(winterload.forecast.lifecycle.retrain_trigger.drift_reference) and logs everything to MLflow, as its own run
`drift_report` in MONITORING_EXPERIMENT: the share of drifted columns, each column's score, the HTML report
and its JSON. A row per day also goes to the result store (DRIFT_LOG). retrain_monitor calls it and adds
its decision to the MAPE trigger.
"""

import datetime as dt
import logging
import tempfile
from pathlib import Path

import pandas as pd

from winterload.forecast.evaluation.drift import measure_drift
from winterload.forecast.lifecycle.champion import ChampionRecord
from winterload.forecast.lifecycle.retrain_trigger import (
    MIN_MONITORING_HOURS,
    DriftDecision,
    DriftTriggerPolicy,
    drift_reference,
)
from winterload.repository.champion import MONITORING_EXPERIMENT, ChampionRepository
from winterload.repository.factory import result_repository
from winterload.repository.results import DRIFT_LOG
from winterload.repository.training_frame import load_dataframe
from winterload.storage import Storage

RUN_NAME = "drift_report"


def report_drift(storage: Storage, registry: ChampionRepository, champion: ChampionRecord | None,
                 current: pd.DataFrame, window: tuple[dt.date, dt.date], test_year_released: bool,
                 reference_date: dt.date, logger: logging.Logger) -> DriftDecision | None:
    """Logs the drift of `current` (the rows of `window`). None if there is no reference to compare with."""
    start, end = window
    reference = drift_reference(start, end, test_year_released, champion.training_end if champion else None)
    if reference is None:
        logger.info("DRIFT: skipped, no champion yet, and the test year is still closed.")
        return None

    frames = [load_dataframe(storage, ref_start, ref_end) for ref_start, ref_end in reference.windows]
    reference_frame = pd.concat(frames, ignore_index=True)
    if len(reference_frame) < MIN_MONITORING_HOURS:
        logger.warning("DRIFT: skipped, reference %s %s has %s row(s), below the minimum of %s.", reference.kind,
                       reference.windows, len(reference_frame), MIN_MONITORING_HOURS)
        return None

    result = measure_drift(reference_frame, current)
    decision = DriftTriggerPolicy().decide(result.drift_share)
    logger.info("DRIFT: %s of %s column(s) drifted against %s (share %.2f, threshold %.2f): %s",
                result.drifted_columns, len(result.columns), reference.kind, result.drift_share,
                decision.threshold_share, "retrain" if decision.retrain else "keep")

    with registry.run(RUN_NAME, MONITORING_EXPERIMENT):
        registry.log(
            params={"reference_date": reference_date, "window_start": start, "window_end": end,
                    "reference_kind": reference.kind,
                    "reference_windows": " ".join(f"{a}..{b}" for a, b in reference.windows),
                    "current_rows": len(current), "reference_rows": len(reference_frame),
                    "retrain": decision.retrain}
                   | {f"method.{column}": drift.method for column, drift in result.columns.items()},
            metrics={"drift_share": result.drift_share, "drifted_columns": result.drifted_columns,
                     "threshold_share": decision.threshold_share}
                    | {f"score.{column}": drift.score for column, drift in result.columns.items()}
                    | {f"drifted.{column}": float(drift.drifted) for column, drift in result.columns.items()},
            datasets={"reference": reference_frame, "current": current},
        )
        registry.tag({"drift_detected": decision.retrain})
        registry.log_dict(result.snapshot.dict(), "drift_report.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift_report.html"
            result.snapshot.save_html(str(path))
            registry.log_artifact(str(path))

    result_repository().append(DRIFT_LOG, pd.DataFrame([
        {"run_at": pd.Timestamp.now(tz="UTC").floor("s"), "reference_date": reference_date, "window_start": start,
         "window_end": end, "reference_kind": reference.kind, "drift_share": result.drift_share,
         "drifted_columns": result.drifted_columns, "retrain": decision.retrain}
        | {f"score.{column}": drift.score for column, drift in result.columns.items()}
    ]))
    return decision
