from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, String
from feast.value_type import ValueType

from winterload.storage import get_storage

# weather observations
location = Entity(name="location", join_keys=["location_id"], value_type=ValueType.STRING)

# energy loads
bidding_zone = Entity(name="bidding_zone", join_keys=["bidding_zone_id"], value_type=ValueType.STRING)

# Read from whichever backend OFFLINE_STORAGE selects -- the same one the backfill pipelines
# (winterload.pipeline.features) write one day at a time to, which is what makes them resumable.
storage = get_storage()

weather_forecast_source = storage.source("weather_forecast")

weather_measurements_source = storage.source("weather_measurements")

energy_load_source = storage.source("energy_load")

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
