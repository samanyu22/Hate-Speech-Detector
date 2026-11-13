import pytest
from src.rewards.reward_function import MultiObjectiveRewardCalculator

def test_reward_accurate_prediction():
    calc = MultiObjectiveRewardCalculator()
    res = calc.calculate_reward(
        predicted_label=1,
        confidence=0.95,
        is_correct=True,
        faithfulness_score=0.90,
        explanation_rating=5,
        context_used=True
    )
    assert res["total_reward"] > 0.70
    assert res["r_error"] == 0.0
    assert res["r_classification"] == 1.0

def test_reward_inaccurate_prediction_penalty():
    calc = MultiObjectiveRewardCalculator()
    res = calc.calculate_reward(
        predicted_label=1,
        confidence=0.90,
        is_correct=False,
        suggested_label=0,
        faithfulness_score=0.40,
        explanation_rating=1,
        context_used=False
    )
    # Confident false positive should incur lambda_err * confidence penalty
    assert res["r_error"] == 0.90
    assert res["r_classification"] == 0.0
    assert res["total_reward"] < 0.50

