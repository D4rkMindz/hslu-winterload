import datetime as dt

import pytest
from winterload.forecast.evaluation.splits import TEST_END, TEST_START
from winterload.forecast.lifecycle.retrain_trigger import RetrainTriggerPolicy, monitoring_window


def test_retrains_when_the_model_loses_its_edge_over_the_baseline():
    decision = RetrainTriggerPolicy().decide(model_mape=0.07, baseline_mape=0.07)

    assert decision.retrain
    assert decision.threshold_mape == pytest.approx(0.063)


def test_keeps_the_model_while_it_beats_the_threshold():
    assert not RetrainTriggerPolicy().decide(model_mape=0.05, baseline_mape=0.07).retrain


def test_exactly_at_the_threshold_does_not_retrain():
    assert not RetrainTriggerPolicy().decide(model_mape=0.09, baseline_mape=0.10).retrain


def test_the_same_mape_is_judged_against_the_season():
    # Monthly baseline MAPE spans 4.6% to 10.8% (docs/proposal.pdf). A flat threshold would give the
    # same answer in both months.
    summer = RetrainTriggerPolicy().decide(model_mape=0.06, baseline_mape=0.046)
    winter = RetrainTriggerPolicy().decide(model_mape=0.06, baseline_mape=0.108)

    assert summer.retrain
    assert not winter.retrain


def test_monitoring_window_opens_a_week_after_the_test_year():
    assert monitoring_window(TEST_END + dt.timedelta(days=7)) == (TEST_END, TEST_END + dt.timedelta(days=7))


def test_monitoring_window_stays_closed_inside_the_test_year():
    assert monitoring_window(TEST_END + dt.timedelta(days=6)) is None
    assert monitoring_window(TEST_END) is None
    assert monitoring_window(TEST_START + dt.timedelta(days=30)) is None
