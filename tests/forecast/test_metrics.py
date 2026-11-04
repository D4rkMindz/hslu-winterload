import numpy as np
import pytest
from winterload.forecast.metrics import mae, mape


def test_mape_is_a_fraction():
    assert mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == pytest.approx(0.10)


def test_mae_is_in_the_target_unit():
    assert mae(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == pytest.approx(15.0)
