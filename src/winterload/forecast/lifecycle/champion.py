"""The champion: two models under one registered name, reviewer point B2/A3 and docs/proposal.pdf,
section 4. `champion_eval` stops at the cutoff of the run that promoted it, so later runs can score
it on hours it has not seen. `champion` is the winner refit on all data, for inference.
winterload.repository.champion stores both.
"""

from dataclasses import dataclass

from winterload.forecast.model.preliminary import PreliminaryModel


@dataclass(frozen=True)
class ChampionRecord:
    evaluation_model: PreliminaryModel
    holdout_mape: float
    promoted_at: str
