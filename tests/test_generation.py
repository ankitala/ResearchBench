from researchbench.generation import estimate_generation_model_calls, parse_matched_score, run_generation_record, score_generation
from researchbench.clients import MockClient


def test_parse_matched_score_accepts_zero_to_five() -> None:
    assert parse_matched_score("**Matched score starts:** 4 **Matched score ends**") == 4.0
    assert parse_matched_score("Reason: ok\nMatched score: 4") == 4.0
    assert parse_matched_score("Matched score: 5.0") == 5.0
    assert parse_matched_score("Matched score: 6") is None
    assert parse_matched_score("no score") is None


def test_score_generation_normalizes_average_matched_score() -> None:
    predictions = [
        {"sample_id": "s1", "final_hypothesis": "hypothesis", "matched_score": 2},
        {"sample_id": "s2", "final_hypothesis": "hypothesis", "matched_score": 4},
    ]
    data = [
        {"sample_id": "s1", "gold_hypothesis": "gold 1"},
        {"sample_id": "s2", "gold_hypothesis": "gold 2"},
    ]
    result = score_generation(predictions, data, MockClient())

    assert result["summary"]["average_score"] == 3.0
    assert result["summary"]["accuracy"] == 0.6


def test_run_generation_accepts_legacy_ture_retrieve_key() -> None:
    record = {
        "sample_id": "s1",
        "research_question": "question",
        "background_survey": "survey",
        "ture_retrieve": [{"title": "Inspiration", "abstract": "abstract"}],
    }
    result = run_generation_record(record, MockClient(), num_mutations=1, num_itr_self_refine=1)

    assert "moosechem_trace" in result
    assert result["moosechem_trace"]["inspiration_results"][0]["inspiration_title"] == "Inspiration"
    assert result["final_hypothesis"]


def test_run_generation_defaults_to_no_background_survey_for_moosechem_reproducibility() -> None:
    record = {
        "sample_id": "s1",
        "research_question": "question",
        "background_survey": "private survey text",
        "gold_inspirations": [{"title": "Inspiration", "abstract": "abstract"}],
    }
    result = run_generation_record(record, MockClient(), num_mutations=1, num_itr_self_refine=1)
    prompt = result["moosechem_trace"]["inspiration_results"][0]["mutations"]["0"]["trace"][0]["prompt"]

    assert "Survey not provided. Please overlook the survey." in prompt
    assert "private survey text" not in prompt
    assert result["moosechem_trace"]["params"]["if_use_background_survey"] is False


def test_estimate_generation_model_calls_default_three_inspirations() -> None:
    record = {
        "sample_id": "s1",
        "gold_inspirations": [
            {"title": "A", "abstract": "a"},
            {"title": "B", "abstract": "b"},
            {"title": "C", "abstract": "c"},
        ],
    }

    assert estimate_generation_model_calls(record) == 105
