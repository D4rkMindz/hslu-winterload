import pytest
from winterload.forecast.target import compute_target_mape


def test_compute_target_mape_default_ten_percent_relative_improvement():
    assert compute_target_mape(0.0615) == pytest.approx(0.0615 * 0.9)


def test_compute_target_mape_relative_not_absolute():
    # A relative (not absolute) reduction: 10% off of a 10% MAPE is 9%, not 8.5%pp. Mixing up
    # relative and absolute reductions is exactly the class of error reviewers flagged in the
    # original proposal's "7.6% -> 6.5% is a 15% reduction" claim, which is actually 14.5%.
    assert compute_target_mape(0.10) == pytest.approx(0.09)


def test_compute_target_mape_custom_relative_improvement():
    assert compute_target_mape(0.076, relative_improvement=0.145) == pytest.approx(0.065, abs=1e-4)
