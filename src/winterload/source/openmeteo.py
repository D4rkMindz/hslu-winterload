# initially copied from https://open-meteo.com/en/docs/meteoswiss-api?hourly=temperature_2m,precipitation,rain,apparent_temperature,dew_point_2m,relative_humidity_2m,snowfall,snow_depth,cloud_cover,cloud_cover_mid,cloud_cover_low,cloud_cover_high,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wet_bulb_temperature_2m,cloud_base,sunshine_duration,snowfall_height,freezing_level_height&time_mode=time_interval&api_mode=historical_forecast&start_date=2022-01-01&end_date=2022-12-31&models=meteoswiss_icon_seamless,meteoswiss_icon_ch1,meteoswiss_icon_ch2&daily=weather_code,temperature_2m_min,temperature_2m_max,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,sunshine_duration,daylight_duration,rain_sum,showers_sum,snowfall_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration
# then rewritten to a class
import logging
from datetime import date, datetime

import openmeteo_requests
from openmeteo_sdk import Model
import pandas as pd
import requests_cache
from retry_requests import retry

# Named weather features used by the model (docs/proposal.pdf, Named features): mean of the
# four cities' temperature, cloud cover and shortwave radiation.
HOURLY_VARIABLES = ["temperature_2m", "cloud_cover", "shortwave_radiation"]

# ECMWF IFS HRES (9km): the only model with a Single Runs archive before April 2026, and used
# consistently across backfill, live inference and the legacy historic source so no step trades
# one model's bias for another's.
MODEL = "ecmwf_ifs"


class OpenMeteo(object):
    HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com"
    SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com"
    FORECAST_URL = "https://api.open-meteo.com"

    def __init__(self, cache_expire_after: int = 10):
        cache_session = requests_cache.CachedSession(".cache", expire_after=cache_expire_after)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_model_name(code: int) -> str:
        debug_items = Model.Model.__dict__.items();
        for name, value in debug_items:
            if value == code:
                return name
        return ""

    def get_weather_backfill(self, lat: float, lon: float, run: datetime) -> dict:
        self.logger.info("Getting backfill weather for run %s at (%s, %s)", run.isoformat(), lat, lon)
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "hourly": HOURLY_VARIABLES,
            "models": MODEL,
            "run": run.strftime("%Y-%m-%dT%H:%M"),
        }
        responses = self.client.weather_api(f"{self.SINGLE_RUNS_URL}/v1/forecast", params=params, timeout=120)
        return self._parse_by_model(responses)

    def get_weather_forecast(self, lat: float, lon: float, forecast_hours: int = 48) -> dict:
        self.logger.info("Getting live weather forecast at (%s,%s)", lat, lon)
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "hourly": HOURLY_VARIABLES,
            "models": MODEL,
            "forecast_hours": forecast_hours,
        }
        responses = self.client.weather_api(f"{self.FORECAST_URL}/v1/forecast", params=params, timeout=120)
        return self._parse_by_model(responses)

    def get_weather_historic(self, lat: float, lon: float, start_date: date, end_date: date) -> dict:
        self.logger.info("Getting historical (short-lead) weather data between %s and %s at (%s,%s)", start_date,
                         end_date, lat, lon)
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "hourly": HOURLY_VARIABLES,
            "models": MODEL,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        responses = self.client.weather_api(f"{self.HISTORICAL_FORECAST_URL}/v1/forecast", params=params, timeout=120)
        return self._parse_by_model(responses)

    def _parse_by_model(self, responses) -> dict:
        result = {}
        for response in responses:
            self.logger.debug("Coordinates: %s°N %s°E", response.Latitude(), response.Longitude())
            self.logger.debug("Elevation: %s m asl", response.Elevation())
            self.logger.debug("Timezone difference to GMT+0: %ss", response.UtcOffsetSeconds())
            self.logger.debug("Model Nº: %s", response.Model())

            hourly = response.Hourly()
            hourly_rows = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(hourly.Interval(), unit="s"),
                    inclusive="left",
                ),
            }
            for i, name in enumerate(HOURLY_VARIABLES):
                hourly_rows[name] = hourly.Variables(i).ValuesAsNumpy()

            result[OpenMeteo.get_model_name(response.Model())] = {"hourly": hourly_rows}

        return result
