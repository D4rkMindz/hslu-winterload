from winterload.forecast.lifecycle.promotion import PromotionPolicy


def test_no_champion_yet_promotes_unconditionally():
    decision = PromotionPolicy().decide(champion_mape=None, challenger_mape=0.10)

    assert decision.promote
    assert decision.champion_mape is None
    assert decision.challenger_mape == 0.10


def test_challenger_beats_champion_is_promoted():
    decision = PromotionPolicy().decide(champion_mape=0.08, challenger_mape=0.07)

    assert decision.promote


def test_challenger_worse_than_champion_is_not_promoted():
    decision = PromotionPolicy().decide(champion_mape=0.07, challenger_mape=0.08)

    assert not decision.promote


def test_tie_keeps_the_current_champion():
    decision = PromotionPolicy().decide(champion_mape=0.08, challenger_mape=0.08)

    assert not decision.promote
