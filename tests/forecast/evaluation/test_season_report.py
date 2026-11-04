import pandas as pd
import pytest
from winterload.forecast.evaluation.season_report import SEASON_BY_MONTH, chronological_forecast, \
    compare_seasonal_models, local_months, report_by_holiday, report_by_season
from winterload.forecast.features.dataset import TARGET_COLUMN


def _six_months() -> pd.DataFrame:
    hours = pd.date_range("2025-01-01", "2025-07-01", freq="h", tz="Europe/Zurich", inclusive="left").tz_convert("UTC")
    frame = pd.DataFrame({"event_timestamp": hours})
    frame["month"] = frame["event_timestamp"].dt.tz_convert("Europe/Zurich").dt.month
    frame["load_lag_168h"] = 100.0
    frame[TARGET_COLUMN] = 100.0
    return frame


def test_every_month_has_a_season():
    assert sorted(SEASON_BY_MONTH) == list(range(1, 13))


def test_report_uses_hand_computed_values_and_leaves_out_missing_seasons():
    scored = pd.DataFrame({
        "event_timestamp": pd.to_datetime(["2025-02-01 12:00", "2025-07-01 12:00"], utc=True),
        "month": [2, 7],
        TARGET_COLUMN: [100.0, 200.0],
        "forecast": [110.0, 190.0],
        "load_lag_168h": [120.0, 200.0],
    })

    winter, summer = report_by_season(scored)

    assert (winter.segment, winter.hours, winter.months) == ("winter", 1, ("2025-02",))
    assert winter.model_mape == pytest.approx(0.10)
    assert winter.model_mae == pytest.approx(10.0)
    assert winter.baseline_mape == pytest.approx(0.20)
    assert winter.baseline_mae == pytest.approx(20.0)
    assert (summer.segment, summer.months) == ("summer", ("2025-07",))
    assert summer.baseline_mape == pytest.approx(0.0)


def test_chronological_forecast_skips_months_without_enough_history():
    scored = chronological_forecast(_six_months())

    assert set(local_months(scored)) == {"2025-05", "2025-06"}


def test_chronological_forecast_never_trains_on_the_month_it_scores():
    frame = _six_months()
    # A label jump that only a model trained on June itself could predict.
    frame.loc[local_months(frame) == "2025-06", TARGET_COLUMN] = 1000.0

    scored = chronological_forecast(frame)

    assert scored.loc[local_months(scored) == "2025-06", "forecast"].max() < 200.0


def test_chronological_forecast_works_across_a_gap():
    frame = _six_months()
    frame = frame.loc[local_months(frame) != "2025-03"].reset_index(drop=True)

    assert set(local_months(chronological_forecast(frame))) == {"2025-06"}


def test_too_little_history_gives_an_empty_report():
    frame = _six_months()
    frame = frame.loc[local_months(frame) == "2025-01"].reset_index(drop=True)

    assert report_by_season(chronological_forecast(frame)) == []


def _holiday_rows(shares: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "event_timestamp": pd.date_range("2025-02-01 12:00", periods=len(shares), freq="D", tz="UTC"),
        "month": 2,
        TARGET_COLUMN: 100.0,
        "forecast": [110.0, 120.0, 130.0][:len(shares)],
        "load_lag_168h": 100.0,
        "holiday_share_ch": shares,
    })


def test_report_by_holiday_orders_segments_with_hand_computed_errors():
    national, regional, other = report_by_holiday(_holiday_rows([1.0, 0.3, 0.0]))

    assert (national.segment, regional.segment, other.segment) == ("national holiday", "regional holiday", "other day")
    assert national.model_mape == pytest.approx(0.10)
    assert regional.model_mape == pytest.approx(0.20)
    assert other.model_mape == pytest.approx(0.30)


def test_report_by_holiday_counts_half_as_national():
    (result,) = report_by_holiday(_holiday_rows([0.5]))

    assert result.segment == "national holiday"


def test_report_by_holiday_leaves_out_empty_segments():
    results = report_by_holiday(_holiday_rows([1.0, 0.0]))

    assert [result.segment for result in results] == ["national holiday", "other day"]


def test_same_season_forecast_needs_enough_rows_of_that_season():
    scored = chronological_forecast(_six_months(), same_season_only=True, min_training_hours=8 * 7 * 24)

    assert set(local_months(scored)) == {"2025-05"}


def test_seasonal_comparison_lists_the_months_both_models_score():
    hours = pd.date_range("2024-01-01", "2025-03-01", freq="h", tz="Europe/Zurich", inclusive="left").tz_convert("UTC")
    frame = pd.DataFrame({"event_timestamp": hours})
    frame["month"] = frame["event_timestamp"].dt.tz_convert("Europe/Zurich").dt.month
    frame["load_lag_168h"] = 100.0
    frame[TARGET_COLUMN] = 100.0

    by_season = {result.season: result for result in compare_seasonal_models(frame)}

    assert by_season["summer"].months == ("2024-08",)
    assert by_season["winter"].months == ("2024-11", "2024-12", "2025-01", "2025-02")
    assert by_season["transition"].months == ("2024-05", "2024-09", "2024-10")
