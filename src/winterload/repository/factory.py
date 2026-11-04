from typing import TYPE_CHECKING

from winterload.config import settings
from winterload.repository.results import ResultRepository
from winterload.storage import get_filesystem

if TYPE_CHECKING:
    from winterload.repository.champion import ChampionRepository


def champion_repository() -> "ChampionRepository":
    # Imported here, so callers of result_repository() don't load MLflow. MLflow pulls in tqdm, which
    # prints a TqdmWarning in every notebook without ipywidgets.
    from winterload.repository.champion import ChampionRepository
    return ChampionRepository(settings.mlflow_tracking_uri)


def result_repository() -> ResultRepository:
    return ResultRepository(settings.results_path, get_filesystem(settings.results_path))
