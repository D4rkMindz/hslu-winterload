import logging
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# pydantic-settings' env_file= below only parses .env into this Settings object -- it doesn't
# touch os.environ. pyarrow and fsspec read their S3 credentials (AWS_*, FSSPEC_S3_*) straight
# from os.environ, not through Settings, so they need this too. Must run before `fsspec` is
# imported by anything downstream of this module (incl. transitively via feast/dask): fsspec
# snapshots its env-var config at first import, so loading .env any later leaves the SeaweedFS
# S3 credentials unset and historical retrieval fails with NoCredentialsError/ACCESS_DENIED.
load_dotenv(PROJECT_ROOT / ".env")

# MLflow retries a failed request 7 times with a growing backoff, so a stage waits minutes before it
# reports that the server is down. 3 retries still ride out a short hiccup.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
# MLflow otherwise attaches its own stderr handler, and Kestra shows its INFO lines as errors. Without it,
# MLflow's logs go through the handlers below (INFO to stdout). Must be set before mlflow is imported.
os.environ.setdefault("MLFLOW_CONFIGURE_LOGGING", "false")

# Configured here, at the module everything else imports `settings` from, so every entry point
# (pipeline CLIs, notebooks) gets a working handler just by importing winterload.config -- without
# this, logger.info/.debug calls anywhere in the app are silently dropped (Python's logging has no
# handler by default). Set at INFO until `settings.log_level` is known below, so a config error
# itself (logged, not printed) is still visible even if LOG_LEVEL ends up invalid.
#
# Kestra (like most log collectors) counts every stderr line as an error, so INFO and DEBUG go to stdout
# and only WARNING and up to stderr. The Kestra flows set LOG_FORMAT=kestra instead: every record then goes
# to stdout as a Kestra log line that carries its level (winterload.log_format), so a WARNING shows as one.
LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")
if LOG_FORMAT == "kestra":
    from winterload.log_format import KestraFormatter

    _kestra_handler = logging.StreamHandler(sys.stdout)
    _kestra_handler.setFormatter(KestraFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_kestra_handler])
    # MLflow's tqdm bars write to stderr, so each step would show up as an error
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
else:
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[_stdout_handler, _stderr_handler],
    )
# warnings.warn() otherwise prints two raw lines to stderr; this logs it as one WARNING on logger py.warnings
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    entsoe_api_key: str = Field(alias="ENTSOE_API_KEY")

    kestra_port: int = Field(default=8080, alias="KESTRA_PORT")
    kestra_ui_user: str = Field(default="", alias="KESTRA_UI_USER")
    kestra_ui_password: str = Field(default="", alias="KESTRA_UI_PASSWORD")

    feast_project: str = Field(alias="FEAST_PROJECT")
    # feast_store_path: str = Field(default="s3://feast/", alias="FEAST_S3_STORE_PATH") # online
    feast_store_path: str = Field(default="data/offline", alias="FEAST_OFFLINE_STORE_PATH")
    gcp_project_id: str = Field(alias="GCP_PROJECT_ID")

    # Where the backfills write and Feast reads (see winterload.storage): "s3" or "bigquery"
    offline_storage: Literal["s3", "bigquery"] = Field(default="s3", alias="OFFLINE_STORAGE")
    bigquery_dataset: str = Field(default="winterload", alias="BIGQUERY_DATASET")
    bigquery_location: str = Field(default="europe-west6", alias="BIGQUERY_LOCATION")

    s3_endpoint: str | None = Field(default=None, alias="S3_ENDPOINT")
    # Where notebooks and monitoring keep generated results, see winterload.repository.results.
    results_path: str = Field(default="s3://winterload/results", alias="RESULTS_PATH")

    # 5001 on the host, because macOS AirPlay occupies 5000. See docker/mlflow/docker-compose.yaml.
    # MLflow itself reads MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD from the environment.
    mlflow_tracking_uri: str = Field(default="http://localhost:5001", alias="MLFLOW_TRACKING_URI")
    # The commit the image was built from (Dockerfile, build arg GIT_SHA). Every MLflow run carries it.
    git_sha: str = Field(default="local", alias="GIT_SHA")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


try:
    settings = Settings()
except ValidationError as e:
    logger.critical("Configuration error:\n%s", e)
    sys.exit(1)

logging.getLogger().setLevel(settings.log_level.upper())
