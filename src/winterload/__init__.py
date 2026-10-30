import datetime as dt
from typing import TypedDict

DEFAULT_START_DATE = dt.date(2022, 1, 1)
DEFAULT_END_DATE = dt.date(2026, 9, 23)


class Coordinate(TypedDict):
    lat: float
    lon: float
    name: str


COORDINATES: list[Coordinate] = [
    {"name": "basel", "lat": 47.559601, "lon": 7.588576},
    {"name": "bern", "lat": 46.947975, "lon": 7.447447},
    {"name": "geneva", "lat": 46.204391, "lon": 6.143158},
    {"name": "zurich", "lat": 47.376888, "lon": 8.541694},
]
