import pytest
from winterload import COORDINATES
from winterload.forecast.features.population import CANTON_POPULATION, NEAREST_CITY, city_population_weights


def test_every_canton_has_a_population_and_a_nearest_city():
    assert len(CANTON_POPULATION) == 26
    assert set(NEAREST_CITY) == set(CANTON_POPULATION)


def test_weights_cover_the_weather_cities():
    assert set(city_population_weights()) == {city["name"] for city in COORDINATES}


def test_weights_sum_to_one_and_zurich_is_largest():
    weights = city_population_weights()

    assert sum(weights.values()) == pytest.approx(1)
    assert max(weights, key=weights.get) == "zurich"
