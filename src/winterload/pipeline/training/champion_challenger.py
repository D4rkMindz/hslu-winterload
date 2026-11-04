"""Weekly champion/challenger retraining, reviewer point B2/A3. The challenger trains on data up to
8 weeks ago. Both models then score the most recent 8 weeks, which neither has seen, and
PromotionPolicy picks the winner. The winner gets refit on all data for serving.
"""

import datetime as dt

import pandas as pd
from winterload.forecast.evaluation.splits import CHAMPION_CHALLENGER_HOLDOUT, MIN_HOLDOUT_HOURS, TEST_END, \
    TEST_START, champion_challenger_holdout, training_windows
from winterload.forecast.lifecycle.promotion import PromotionPolicy
from winterload.forecast.model.preliminary import PreliminaryModel
from winterload.pipeline.training import TrainingPipeline
from winterload.pipeline.training.final_test_guard import MARKER_KEY as FINAL_TEST_MARKER_KEY
from winterload.repository.factory import champion_repository
from winterload.repository.training_frame import load_dataframe


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

        holdout_frame = load_dataframe(self.storage, holdout_start, holdout_end)
        if len(holdout_frame) < MIN_HOLDOUT_HOURS:
            self.logger.warning(
                "CHAMPION/CHALLENGER: skipped, holdout %s..%s has %s row(s), below the minimum of %s.",
                holdout_start, holdout_end, len(holdout_frame), MIN_HOLDOUT_HOURS,
            )
            return

        challenger = PreliminaryModel().fit(self._training_frame(holdout_start))
        challenger_mape = challenger.evaluate(holdout_frame)

        registry = champion_repository()
        champion = registry.load_champion()
        champion_mape = champion.evaluation_model.evaluate(holdout_frame) if champion else None

        decision = PromotionPolicy().decide(champion_mape, challenger_mape)
        self.logger.info(
            "CHAMPION/CHALLENGER: holdout %s..%s (%s row(s)), champion MAPE=%s, challenger MAPE=%.4f%%: %s",
            holdout_start, holdout_end, len(holdout_frame),
            f"{champion_mape * 100:.4f}%" if champion_mape is not None else "none yet",
            challenger_mape * 100, "promoted" if decision.promote else "kept champion",
        )

        # PreliminaryModel has no settings yet, so a fresh instance refits either winner.
        serving_model = PreliminaryModel().fit(self._training_frame(holdout_end))
        metrics = {"challenger_mape": challenger_mape, "holdout_rows": len(holdout_frame)}
        if champion_mape is not None:
            metrics["champion_mape"] = champion_mape
        with registry.run(self.name):
            registry.log(
                params={"reference_date": reference_date, "holdout_start": holdout_start,
                        "promoted": decision.promote} | challenger.estimator.get_params(),
                metrics=metrics,
            )
            if decision.promote:
                registry.promote(challenger, serving_model, challenger_mape, dt.datetime.now(dt.UTC))
            else:
                registry.replace_serving_model(serving_model)

    def _training_frame(self, end: dt.date) -> pd.DataFrame:
        released = self.storage.read_marker(FINAL_TEST_MARKER_KEY) is not None
        frames = [load_dataframe(self.storage, start, stop) for start, stop in training_windows(end, released)]
        return pd.concat(frames, ignore_index=True)
