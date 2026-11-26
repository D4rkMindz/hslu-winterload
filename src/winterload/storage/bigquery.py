"""BigQuery tables in GCP_PROJECT_ID.BIGQUERY_DATASET: one table per feature table, partitioned by
`backfill_day` (the day a pipeline wrote the row for) so a day can be checked and replaced on its
own, read by Feast's `bigquery` offline store (feature_store.bigquery.yaml).
"""

import datetime as dt
import json
import logging
from functools import cached_property
from typing import Any

import pandas as pd
from feast import BigQuerySource
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from winterload.config import settings
from winterload.source.openmeteo import HOURLY_VARIABLES
from winterload.storage.base import Storage

logger = logging.getLogger(__name__)

PARTITION_COLUMN = "backfill_day"
MARKERS_TABLE = "_markers"

_weather_fields = [bigquery.SchemaField(name, "FLOAT64") for name in HOURLY_VARIABLES]

# Created up front by `setup()` rather than inferred on the first write: `feast apply` checks that
# every BigQuerySource's table exists, so it would fail on a fresh dataset otherwise. The columns
# match what each backfill pipeline writes (its FEATURE_COLUMNS / feature_columns()).
SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    "energy_load": [
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("bidding_zone_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("quantity", "FLOAT64"),
        bigquery.SchemaField("unit", "STRING"),
    ],
    "weather_forecast": [
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model", "STRING"),
        *_weather_fields,
        bigquery.SchemaField("run_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("lead_time_hours", "FLOAT64"),
    ],
    "weather_measurements": [
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model", "STRING"),
        *_weather_fields,
    ],
}

# The append-only tables of winterload.repository.forecast. No backfill_day: a run never replaces a day, it adds
# rows. Each table is partitioned by the day of the column in APPEND_PARTITIONS.
APPEND_SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    "winterload_forecasts": [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("prediction_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("model_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_role", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("result", "FLOAT64"),
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    ],
    "winterload_forecast_runs": [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("started_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("completed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("target_day", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("run_type", "STRING"),
        bigquery.SchemaField("kestra_execution_id", "STRING"),
    ],
}
APPEND_PARTITIONS = {"winterload_forecasts": "prediction_time", "winterload_forecast_runs": "started_at"}
LATEST_FORECASTS_VIEW = "winterload_forecasts_latest"


class BigQueryStorage(Storage):
    @cached_property
    def client(self) -> bigquery.Client:
        # Created on first use, so definitions.py can build its sources without credentials.
        # Authenticates via Application Default Credentials: `gcloud auth application-default
        # login` locally, or GOOGLE_APPLICATION_CREDENTIALS pointing at a service account key file.
        return bigquery.Client(project=settings.gcp_project_id, location=settings.bigquery_location)

    @staticmethod
    def table_id(table: str) -> str:
        return f"{settings.gcp_project_id}.{settings.bigquery_dataset}.{table}"

    @staticmethod
    def schema(table: str) -> list[bigquery.SchemaField]:
        return [*SCHEMAS[table], bigquery.SchemaField(PARTITION_COLUMN, "DATE", mode="REQUIRED")]

    def day_exists(self, table: str, target_day: dt.date) -> bool:
        query = f"SELECT COUNT(*) AS n FROM `{self.table_id(table)}` WHERE {PARTITION_COLUMN} = @target_day"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("target_day", "DATE", target_day),
        ])
        try:
            row = next(iter(self.client.query(query, job_config=job_config).result()))
        except NotFound:
            return False
        return row["n"] > 0

    def write_day(self, table: str, df: pd.DataFrame, target_day: dt.date) -> str:
        # Loading into the day's partition ($YYYYMMDD) with WRITE_TRUNCATE replaces exactly that
        # day in one atomic job -- the same overwrite semantics as rewriting the day's Parquet file.
        destination = f"{self.table_id(table)}${target_day:%Y%m%d}"
        job_config = bigquery.LoadJobConfig(
            schema=self.schema(table),
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
        )
        df = df.assign(**{PARTITION_COLUMN: target_day})
        self.client.load_table_from_dataframe(df, destination, job_config=job_config).result()
        return destination

    def append(self, table: str, df: pd.DataFrame) -> str:
        job_config = bigquery.LoadJobConfig(
            schema=APPEND_SCHEMAS[table],
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
        )
        self.client.load_table_from_dataframe(df, self.table_id(table), job_config=job_config).result()
        return self.table_id(table)

    def _latest_forecasts_sql(self) -> str:
        """The newest run's result per role, model version and hour. After a promotion champion and challenger
        share a version, so the role stays in the key. Hours a run skipped have no row, so an older run's result
        stays visible for them."""
        return f"""
            CREATE OR REPLACE VIEW `{self.table_id(LATEST_FORECASTS_VIEW)}` AS
            SELECT f.id, f.model_id, f.model_role, f.prediction_time, f.result, f.run_id, r.started_at
            FROM `{self.table_id("winterload_forecasts")}` f
            JOIN `{self.table_id("winterload_forecast_runs")}` r ON f.run_id = r.id
            WHERE f.result IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY f.model_role, f.model_id, f.prediction_time ORDER BY r.started_at DESC) = 1
        """

    def source(self, table: str, model: str | None = None) -> BigQuerySource:
        if model is None:
            return BigQuerySource(name=table, table=self.table_id(table), timestamp_field="event_timestamp")
        query = f"SELECT * FROM `{self.table_id(table)}` WHERE model = '{model}'"
        return BigQuerySource(name=table, query=query, timestamp_field="event_timestamp")

    def read_range(self, table: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        query = f"""
            SELECT * EXCEPT({PARTITION_COLUMN}) FROM `{self.table_id(table)}`
            WHERE event_timestamp >= @start AND event_timestamp < @end
            ORDER BY event_timestamp
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", dt.datetime.combine(start, dt.time.min)),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", dt.datetime.combine(end, dt.time.min)),
        ])
        return self.client.query(query, job_config=job_config).result().to_dataframe()

    def read_appended(self, table: str, column: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        query = f"SELECT * FROM `{self.table_id(table)}` WHERE {column} >= @start AND {column} < @end"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", end),
        ])
        try:
            return self.client.query(query, job_config=job_config).result().to_dataframe()
        except NotFound:
            return pd.DataFrame()

    def read_marker(self, key: str) -> dict[str, Any] | None:
        query = f"SELECT data FROM `{self.table_id(MARKERS_TABLE)}` WHERE key = @key ORDER BY updated_at DESC LIMIT 1"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("key", "STRING", key),
        ])
        try:
            rows = list(self.client.query(query, job_config=job_config).result())
        except NotFound:
            return None
        return json.loads(rows[0]["data"]) if rows else None

    def write_marker(self, key: str, data: dict[str, Any]) -> None:
        # Append-only: read_marker always takes the most recent row for `key`, so a write never
        # needs to delete the previous one -- same single-writer assumption as the rest of this
        # backend (write_day replaces a whole partition instead of upserting rows for the same
        # reason).
        row = {"key": key, "data": json.dumps(data, default=str), "updated_at": dt.datetime.now(dt.UTC)}
        errors = self.client.insert_rows_json(self.table_id(MARKERS_TABLE), [row])
        if errors:
            raise RuntimeError(f"Failed to write marker {key!r}: {errors}")

    def setup(self) -> None:
        """Creates the dataset and the empty, day-partitioned feature tables, if they don't exist yet."""
        dataset = bigquery.Dataset(f"{settings.gcp_project_id}.{settings.bigquery_dataset}")
        dataset.location = settings.bigquery_location
        self.client.create_dataset(dataset, exists_ok=True)
        for table in SCHEMAS:
            bq_table = bigquery.Table(self.table_id(table), schema=self.schema(table))
            bq_table.time_partitioning = bigquery.TimePartitioning(field=PARTITION_COLUMN)
            self.client.create_table(bq_table, exists_ok=True)
        markers_table = bigquery.Table(self.table_id(MARKERS_TABLE), schema=[
            bigquery.SchemaField("key", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("data", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        ])
        self.client.create_table(markers_table, exists_ok=True)
        for table, schema in APPEND_SCHEMAS.items():
            bq_table = bigquery.Table(self.table_id(table), schema=schema)
            bq_table.time_partitioning = bigquery.TimePartitioning(field=APPEND_PARTITIONS[table])
            self.client.create_table(bq_table, exists_ok=True)
        runs = self.table_id("winterload_forecast_runs")
        # Tables from before run_type existed: add the columns, and mark their earlier rows as manual test runs.
        self.client.query(f"ALTER TABLE `{runs}` ADD COLUMN IF NOT EXISTS run_type STRING, "
                          f"ADD COLUMN IF NOT EXISTS kestra_execution_id STRING").result()
        self.client.query(f"UPDATE `{runs}` SET run_type = 'manual' WHERE run_type IS NULL").result()
        self.client.query(self._latest_forecasts_sql()).result()
        logger.info("Dataset %r ready with tables %s and view %s", dataset.dataset_id,
                    ", ".join([*SCHEMAS, MARKERS_TABLE, *APPEND_SCHEMAS]), LATEST_FORECASTS_VIEW)
