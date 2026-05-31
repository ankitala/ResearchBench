from researchbench.clients import MockClient
from researchbench.ranking import parse_selection, run_ranking_record, score_ranking


def test_parse_selection_handles_strict_and_loose_outputs() -> None:
    assert parse_selection("**Selection of research hypothesis candidate**: candidate 2") == 2
    assert parse_selection("I choose candidate 1 because it is clearer.") == 1
    assert parse_selection("no valid selection") is None


def test_ranking_order_reversal_rank_threshold_and_position_bias_stats() -> None:
    record = {
        "sample_id": "sample/1",
        "research_question": "question",
        "gold_hypothesis": "gold",
        "fake_negative_hypotheses": [f"fake {idx}" for idx in range(5)],
        "model_negative_hypotheses": [f"model {idx}" for idx in range(10)],
    }
    pred = run_ranking_record(record, MockClient(), order="both")

    assert pred["orders"]["res"]["rank"] == 16
    assert pred["orders"]["res"]["gold_wins"] is True
    assert pred["orders"]["fan_1_res"]["rank"] == 1
    assert pred["orders"]["fan_1_res"]["gold_wins"] is False

    result = score_ranking([pred])
    assert result["summary"]["overall_accuracy"] == 0.5
    assert result["summary"]["sum_counts"] == {1: 1}
    assert result["summary"]["pair_counts"] == {"self_contradictory": 15}
