from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String
from feast.value_type import ValueType

from winterload.config import settings

# weather observations
location = Entity(name="location", join_keys=["location_id"], value_type=ValueType.STRING)

# energy loads
bidding_zone = Entity(name="bidding_zone", join_keys=["bidding_zone_id"], value_type=ValueType.STRING)

weather_forecast_source = FileSource(
    # One Parquet file per day (see pipeline.weather_backfill), read as a single dataset -- this
    # is what makes the backfill resumable instead of risking a multi-hour single-file rewrite.
    path=f"{settings.feast_store_path}/weather_forecast",
    timestamp_field="event_timestamp",
    s3_endpoint_override=settings.s3_endpoint,
)

weather_measurements_source = FileSource(
    # One Parquet file per day (see pipeline.weather_measurements_backfill), same resumable
    # layout as weather_forecast_source.
    path=f"{settings.feast_store_path}/weather_measurements",
    timestamp_field="event_timestamp",
    s3_endpoint_override=settings.s3_endpoint,
)

energy_load_source = FileSource(
    # One Parquet file per day (see pipeline.energy_load_backfill), same resumable layout as
    # weather_forecast_source.
    path=f"{settings.feast_store_path}/energy_load",
    timestamp_field="event_timestamp",
    s3_endpoint_override=settings.s3_endpoint,
)

# Shared by weather_forecast_stats and weather_measurements_stats; matches
# winterload.source.openmeteo.HOURLY_VARIABLES so forecast and measurement rows are directly
# comparable. weather_forecast_stats holds the 16-39h-ahead forecast reconstructed per-run via
# OpenMeteo.get_weather_backfill() (training/backtest) or read live via get_weather_forecast()
# (production) -- never the observed value, which would leak the future into training.
weather_schema = [
    Field(name="temperature_2m", dtype=Float32),
    Field(name="cloud_cover", dtype=Float32),
    Field(name="shortwave_radiation", dtype=Float32),
]

weather_forecast_stats = FeatureView(
    name="weather_forecast_stats",
    entities=[location],
    ttl=timedelta(days=1),
    schema=weather_schema,
    source=weather_forecast_source,
)

weather_measurements_stats = FeatureView(
    name="weather_measurements_stats",
    entities=[location],
    ttl=timedelta(days=1),
    schema=weather_schema,
    source=weather_measurements_source,
)

energy_load_stats = FeatureView(
    name="energy_load_stats",
    entities=[bidding_zone],
    ttl=timedelta(days=1),
    schema=[
        Field(name="quantity", dtype=Float32),
        Field(name="unit", dtype=String),
    ],
    source=energy_load_source,
)
