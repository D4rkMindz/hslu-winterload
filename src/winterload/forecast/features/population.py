"""Swiss population figures for national aggregates, reviewer points A6 and E2. Nobody publishes
load per canton, so population stands in for demand.
"""

# Residents per canton on 30 September 2025, https://de.wikipedia.org/wiki/Kanton_(Schweiz).
CANTON_POPULATION = {
    "ZH": 1632918, "BE": 1069074, "LU": 441500, "UR": 38581, "SZ": 170072, "OW": 39879, "NW": 45664,
    "GL": 42451, "ZG": 135018, "FR": 350918, "SO": 291583, "BS": 204157, "BL": 303872, "SH": 90087,
    "AR": 57221, "AI": 16890, "SG": 544040, "GR": 207400, "AG": 743768, "TG": 302806, "TI": 359783,
    "VD": 863650, "VS": 376113, "NE": 180365, "GE": 537084, "JU": 82231,
}

# The weather city closest to each canton's capital.
NEAREST_CITY = {
    "ZH": "zurich", "LU": "zurich", "UR": "zurich", "SZ": "zurich", "OW": "zurich", "NW": "zurich",
    "GL": "zurich", "ZG": "zurich", "SH": "zurich", "AR": "zurich", "AI": "zurich", "SG": "zurich",
    "GR": "zurich", "AG": "zurich", "TG": "zurich", "TI": "zurich",
    "BE": "bern", "FR": "bern", "SO": "bern", "VS": "bern", "NE": "bern",
    "BS": "basel", "BL": "basel", "JU": "basel",
    "VD": "geneva", "GE": "geneva",
}


def city_population_weights() -> dict[str, float]:
    """Each weather city's share of the Swiss population, counting every canton to its nearest city."""
    total = sum(CANTON_POPULATION.values())
    weights: dict[str, float] = {}
    for canton, population in CANTON_POPULATION.items():
        city = NEAREST_CITY[canton]
        weights[city] = weights.get(city, 0.0) + population / total
    return weights
