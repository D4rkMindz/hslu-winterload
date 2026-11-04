"""Generated results: logs and tables from notebooks and monitoring, as CSV objects under one root.
They stay out of git. The root is SeaweedFS by default, see RESULTS_PATH in .env.example.

The `filesystem` is any fsspec filesystem, handed in by winterload.repository.factory. This module
must not import fsspec: fsspec reads its S3 settings once, at first import, and the config module
has to load .env before that.
"""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

PUBLICATION_LOG = "backtest/publication_log"
WEATHER_RUN_LOG = "backtest/weather_run_log"
LOAD_SNAPSHOTS = "backtest/load_snapshots"
REVISION_LOG = "backtest/revision_log"
REPLAY_LOG = "backtest/replay_log"
SEASON_REPORT = "backtest/season_report"
WEATHER_RUN_CHOICE = "benchmark/weather_run_choice"
WEATHER_ARCHIVE_MAPE = "benchmark/weather_archive_mape"
FEATURE_GROUPS = "training/feature_groups"
HOLIDAY_ERRORS = "training/holiday_errors"
MODEL_SETTINGS = "training/model_settings"
SEASONAL_MODELS = "training/seasonal_models"
WEATHER_WEIGHTS = "training/weather_weights"
ORDER_NOISE = "training/order_noise"


class ResultRepository:
    def __init__(self, root: str, filesystem: "AbstractFileSystem") -> None:
        self._root = root.rstrip("/")
        self._filesystem = filesystem

    def path(self, name: str) -> str:
        return f"{self._root}/{name}.csv"

    def read(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        """The result called `name`. Empty if nothing has written it yet."""
        path = self.path(name)
        # s3fs caches listings, and another process may have written since.
        self._filesystem.invalidate_cache()
        if not self._filesystem.exists(path):
            return pd.DataFrame()
        with self._filesystem.open(path, "r") as file:
            return pd.read_csv(file, parse_dates=parse_dates)

    def write(self, name: str, rows: pd.DataFrame) -> str:
        """Replaces the result called `name`. Returns where it was written to, for logging."""
        path = self.path(name)
        self._filesystem.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        with self._filesystem.open(path, "w") as file:
            rows.to_csv(file, index=False)
        return path

    def append(self, name: str, rows: pd.DataFrame) -> str:
        """Adds `rows` to the result called `name`. S3 objects can't grow, so this rewrites the object."""
        return self.write(name, pd.concat([self.read(name), rows], ignore_index=True))
