"""The LightGBM settings that live retraining uses, as config-as-code.

`-p training.tuning` (Kestra flow `tuning`) searches them and logs the result to MLflow, including a
`tuned_settings.py` artifact with these two lines filled in. Copy that artifact over this block and commit
it. The tuning run ID stays next to the settings: champion_challenger logs it, and `-p training.final_test`
checks that run in MLflow (gate and identical settings) before it opens the test year.

Both stay None until the first tuning run is committed. Live retraining then uses PRELIMINARY_SETTINGS,
and final_test refuses to run.
"""

from winterload.forecast.model.preliminary import PRELIMINARY_SETTINGS, LightGBMSettings

TUNED_SETTINGS: LightGBMSettings | None = None
TUNING_RUN_ID: str | None = None


def live_settings() -> tuple[LightGBMSettings, str]:
    """The settings live retraining fits with, and where they come from: the tuning run ID, or
    "preliminary" while nothing is committed."""
    if TUNED_SETTINGS is None or TUNING_RUN_ID is None:
        return PRELIMINARY_SETTINGS, "preliminary"
    return TUNED_SETTINGS, TUNING_RUN_ID
