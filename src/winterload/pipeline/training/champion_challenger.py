"""Weekly champion/challenger retraining, reviewer point B2/A3. The challenger trains on data up to
8 weeks ago. Both models then score the most recent 8 weeks, which neither has seen, and
PromotionPolicy picks the winner. The winner gets refit on all data for serving.

The challenger fits with the committed TUNED_SETTINGS (winterload.forecast.model.tuned), or with LightGBM's
defaults while none are committed. The weekly flow and the monitoring trigger can both start this stage on
the same day, so it skips when another comparison finished in the last 24 hours.
"""

import datetime as dt

import pandas as pd

from winterload.forecast.evaluation.splits import (
    CHAMPION_CHALLENGER_HOLDOUT,
    MIN_HOLDOUT_HOURS,
    TEST_END,
    TEST_START,
    champion_challenger_holdout,
    training_windows,
)
from winterload.forecast.lifecycle.promotion import PromotionPolicy
from winterload.forecast.model.preliminary import PreliminaryModel
from winterload.forecast.model.tuned import live_settings
from winterload.pipeline.training import TrainingPipeline
from winterload.pipeline.training.final_test_guard import MARKER_KEY as FINAL_TEST_MARKER_KEY
from winterload.repository.champion import TRAINING_EXPERIMENT
from winterload.repository.factory import champion_repository
from winterload.repository.training_frame import load_dataframe

# One comparison per day is enough: a second one would score the same holdout.
MIN_RUN_INTERVAL = dt.timedelta(hours=24)


class ChampionChallengerPipeline(TrainingPipeline):
    name = "champion_challenger"
    order: int = 220

    def run(self, reference_date: dt.date | None = None) -> None:
        reference_date = reference_date or dt.datetime.now(dt.UTC).date()

        holdout = champion_challenger_holdout(reference_date)
        if holdout is None:
            self.logger.info(
                "CHAMPION/CHALLENGER: skipped, an 8-week holdout from %s would overlap the final test "
                "year (%s..%s). Earliest valid run: %s.",
                reference_date, TEST_START, TEST_END, TEST_END + CHAMPION_CHALLENGER_HOLDOUT,
            )
            return
        holdout_start, holdout_end = holdout

        registry = champion_repository()
        recent = registry.finished_run_since(self.name, TRAINING_EXPERIMENT, dt.datetime.now(dt.UTC) - MIN_RUN_INTERVAL)
        if recent is not None:
            self.logger.info("CHAMPION/CHALLENGER: skipped, run %s already compared the models in the last %s.",
                             recent, MIN_RUN_INTERVAL)
            return

        holdout_frame = load_dataframe(self.storage, holdout_start, holdout_end)
        if len(holdout_frame) < MIN_HOLDOUT_HOURS:
            self.logger.warning(
                "CHAMPION/CHALLENGER: skipped, holdout %s..%s has %s row(s), below the minimum of %s.",
                holdout_start, holdout_end, len(holdout_frame), MIN_HOLDOUT_HOURS,
            )
            return

        settings, settings_source = live_settings()
        challenger_train = self._training_frame(holdout_start)
        challenger = PreliminaryModel(settings).fit(challenger_train)
        challenger_mape = challenger.evaluate(holdout_frame)

        champion = registry.load_champion()
        champion_mape = champion.evaluation_model.evaluate(holdout_frame) if champion else None

        decision = PromotionPolicy().decide(champion_mape, challenger_mape)
        self.logger.info(
            "CHAMPION/CHALLENGER: holdout %s..%s (%s row(s)), champion MAPE=%s, challenger MAPE=%.4f%% "
            "(settings: %s): %s",
            holdout_start, holdout_end, len(holdout_frame),
            f"{champion_mape * 100:.4f}%" if champion_mape is not None else "none yet",
            challenger_mape * 100, settings_source, "promoted" if decision.promote else "kept champion",
        )

        # The winner refits on all data with its own settings: a kept champion may predate the tuned ones.
        winner = challenger if decision.promote else champion.evaluation_model
        serving_train = self._training_frame(holdout_end)
        serving_model = winner.unfitted_copy().fit(serving_train)
        metrics = {"challenger_mape": challenger_mape, "holdout_rows": len(holdout_frame)}
        if champion_mape is not None:
            metrics["champion_mape"] = champion_mape
        with registry.run(self.name, TRAINING_EXPERIMENT):
            registry.log(
                params={"reference_date": reference_date, "holdout_start": holdout_start,
                        "promoted": decision.promote, "settings_source": settings_source}
                       | challenger.estimator.get_params(),
                metrics=metrics,
                datasets={"challenger_train": challenger_train, "holdout": holdout_frame,
                          "serving_train": serving_train},
            )
            if decision.promote:
                registry.promote(challenger, serving_model, challenger_mape, dt.datetime.now(dt.UTC),
                                 training_end=holdout_start)
            else:
                registry.replace_serving_model(serving_model)

    def _training_frame(self, end: dt.date) -> pd.DataFrame:
        released = self.storage.read_marker(FINAL_TEST_MARKER_KEY) is not None
        frames = [load_dataframe(self.storage, start, stop) for start, stop in training_windows(end, released)]
        return pd.concat(frames, ignore_index=True)
