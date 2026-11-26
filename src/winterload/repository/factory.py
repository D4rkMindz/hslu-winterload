from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem

from winterload.config import PROJECT_ROOT, settings
from winterload.repository.results import ResultRepository
from winterload.storage import get_filesystem

if TYPE_CHECKING:
    from winterload.repository.champion import ChampionRepository


def champion_repository() -> ChampionRepository:
    # Imported here, so callers of result_repository() don't load MLflow. MLflow pulls in tqdm, which
    # prints a TqdmWarning in every notebook without ipywidgets.
    from winterload.repository.champion import ChampionRepository
    return ChampionRepository(settings.mlflow_tracking_uri, settings.git_sha)


def result_repository() -> ResultRepository:
    return ResultRepository(settings.results_path, get_filesystem(settings.results_path))


def evidence_repository() -> ResultRepository:
    """The run evidence, which lives in git under docs/runs."""
    return ResultRepository(str(PROJECT_ROOT / "docs" / "runs"), LocalFileSystem())
