"""Decides whether a retrained challenger replaces the champion, reviewer point B2/A3. It compares
two MAPEs and ignores how they were measured. The caller must score both models on the same hours
that neither trained on: see winterload.forecast.evaluation.splits.champion_challenger_holdout.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    champion_mape: float | None
    challenger_mape: float


class PromotionPolicy:
    def decide(self, champion_mape: float | None, challenger_mape: float) -> PromotionDecision:
        """`champion_mape` is None if no champion exists yet, and the challenger then becomes the
        first one. Otherwise the challenger must beat the champion, and a tie keeps the champion."""
        promote = champion_mape is None or challenger_mape < champion_mape
        return PromotionDecision(promote=promote, champion_mape=champion_mape, challenger_mape=challenger_mape)
