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
        *_weather_fields,
        bigquery.SchemaField("run_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("lead_time_hours", "FLOAT64"),
    ],
    "weather_measurements": [
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
        *_weather_fields,
    ],
}


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

    def source(self, table: str) -> BigQuerySource:
        return BigQuerySource(name=table, table=self.table_id(table), timestamp_field="event_timestamp")

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
        logger.info("Dataset %r ready with tables %s", dataset.dataset_id, ", ".join([*SCHEMAS, MARKERS_TABLE]))
