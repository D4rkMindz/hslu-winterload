# initially copied from https://open-meteo.com/en/docs/meteoswiss-api?hourly=temperature_2m,precipitation,rain,apparent_temperature,dew_point_2m,relative_humidity_2m,snowfall,snow_depth,cloud_cover,cloud_cover_mid,cloud_cover_low,cloud_cover_high,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wet_bulb_temperature_2m,cloud_base,sunshine_duration,snowfall_height,freezing_level_height&time_mode=time_interval&api_mode=historical_forecast&start_date=2022-01-01&end_date=2022-12-31&models=meteoswiss_icon_seamless,meteoswiss_icon_ch1,meteoswiss_icon_ch2&daily=weather_code,temperature_2m_min,temperature_2m_max,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,sunshine_duration,daylight_duration,rain_sum,showers_sum,snowfall_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration
# then rewritten to a class
from datetime import date

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

class OpenMeteo(object):
    BASE_URL = "https://historical-forecast-api.open-meteo.com"
    BASE_PARAMS = {
        # DONT YOU DARE TOUCHING THE ORDER OF THOSE!
        "daily": ["weather_code", "temperature_2m_min", "temperature_2m_max", "apparent_temperature_max",
                  "apparent_temperature_min", "sunrise", "sunset", "sunshine_duration", "daylight_duration", "rain_sum",
                  "showers_sum", "snowfall_sum", "precipitation_sum", "precipitation_hours",
                  "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max",
                  "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
        "hourly": ["temperature_2m", "precipitation", "rain", "apparent_temperature", "dew_point_2m",
                   "relative_humidity_2m", "snowfall", "snow_depth", "cloud_cover", "cloud_cover_mid",
                   "cloud_cover_low",
                   "cloud_cover_high", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                   "wet_bulb_temperature_2m", "cloud_base", "sunshine_duration", "snowfall_height",
                   "freezing_level_height"],
        # "models": ["meteoswiss_icon_seamless", "meteoswiss_icon_ch1", "meteoswiss_icon_ch2"],
    }

    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=10)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.log = print

    def get_weather_historic(self, lat: float, lon: float, start_date: date, end_date: date):
        self.log(f"Getting historical weather data between {start_date} and {end_date} at ({lat},{lon})")
        params = self.BASE_PARAMS.copy()
        params["latitude"] = str(lat)
        params["longitude"] = str(lon)
        params["start_date"] = start_date.strftime("%Y-%m-%d")
        params["end_date"] = end_date.strftime("%Y-%m-%d")
        responses = self.client.weather_api(f"{self.BASE_URL}/v1/forecast", params=params, timeout=120)

        result = {}
        for response in responses:
            self.log(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
            self.log(f"Elevation: {response.Elevation()} m asl")
            self.log(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
            self.log(f"Model Nº: {response.Model()}")

            hourly = response.Hourly()
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "precipitation": hourly.Variables(1).ValuesAsNumpy(),
                "rain": hourly.Variables(2).ValuesAsNumpy(),
                "apparent_temperature": hourly.Variables(3).ValuesAsNumpy(),
                "dew_point_2m": hourly.Variables(4).ValuesAsNumpy(),
                "relative_humidity_2m": hourly.Variables(5).ValuesAsNumpy(),
                "snowfall": hourly.Variables(6).ValuesAsNumpy(),
                "snow_depth": hourly.Variables(7).ValuesAsNumpy(),
                "cloud_cover": hourly.Variables(8).ValuesAsNumpy(),
                "cloud_cover_mid": hourly.Variables(9).ValuesAsNumpy(),
                "cloud_cover_low": hourly.Variables(10).ValuesAsNumpy(),
                "cloud_cover_high": hourly.Variables(11).ValuesAsNumpy(),
                "surface_pressure": hourly.Variables(12).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(13).ValuesAsNumpy(),
                "wind_direction_10m": hourly.Variables(14).ValuesAsNumpy(),
                "wind_gusts_10m": hourly.Variables(15).ValuesAsNumpy(),
                "wet_bulb_temperature_2m": hourly.Variables(16).ValuesAsNumpy(),
                "cloud_base": hourly.Variables(17).ValuesAsNumpy(),
                "sunshine_duration": hourly.Variables(18).ValuesAsNumpy(),
                "snowfall_height": hourly.Variables(19).ValuesAsNumpy(),
                "freezing_level_height": hourly.Variables(20).ValuesAsNumpy(),
            }

            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "sunrise": pd.to_datetime(daily.Variables(5).ValuesInt64AsNumpy(), unit="s", utc=True),
                "sunset": pd.to_datetime(daily.Variables(6).ValuesInt64AsNumpy(), unit="s", utc=True),
                "weather_code": daily.Variables(0).ValuesAsNumpy(),
                "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
                "temperature_2m_max": daily.Variables(2).ValuesAsNumpy(),
                "apparent_temperature_max": daily.Variables(3).ValuesAsNumpy(),
                "apparent_temperature_min": daily.Variables(4).ValuesAsNumpy(),
                "sunshine_duration": daily.Variables(7).ValuesAsNumpy(),
                "daylight_duration": daily.Variables(8).ValuesAsNumpy(),
                "rain_sum": daily.Variables(9).ValuesAsNumpy(),
                "showers_sum": daily.Variables(10).ValuesAsNumpy(),
                "snowfall_sum": daily.Variables(11).ValuesAsNumpy(),
                "precipitation_sum": daily.Variables(12).ValuesAsNumpy(),
                "precipitation_hours": daily.Variables(13).ValuesAsNumpy(),
                "precipitation_probability_max": daily.Variables(14).ValuesAsNumpy(),
                "wind_speed_10m_max": daily.Variables(15).ValuesAsNumpy(),
                "wind_gusts_10m_max": daily.Variables(16).ValuesAsNumpy(),
                "wind_direction_10m_dominant": daily.Variables(17).ValuesAsNumpy(),
                "shortwave_radiation_sum": daily.Variables(18).ValuesAsNumpy(),
                "et0_fao_evapotranspiration": daily.Variables(19).ValuesAsNumpy(),
            }

            result[response.Model()] = {
                "hourly": hourly_data,
                "daily": daily_data,
            }

        return result
