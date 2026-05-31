from researchbench.clients import MockClient
from researchbench.retrieve import parse_retrieve_response, run_retrieve_record, score_retrieve


def test_parse_retrieve_response_extracts_titles_and_reasons() -> None:
    parsed = parse_retrieve_response(
        "Title: First paper\n"
        "Reason: useful bridge.\n"
        "Title: Second paper\n"
        "Reason: complementary mechanism."
    )
    assert parsed == [
        {"title": "First paper", "reason": "useful bridge."},
        {"title": "Second paper", "reason": "complementary mechanism."},
    ]


def test_parse_retrieve_response_extracts_moosechem_title_markers() -> None:
    parsed = parse_retrieve_response(
        "**Title 1 starts:** First paper **Title 1 ends**\n"
        "**Title 2 starts:** Second paper **Title 2 ends**"
    )
    assert parsed == [
        {"title": "First paper", "reason": ""},
        {"title": "Second paper", "reason": ""},
    ]


def test_mock_retrieve_two_rounds_keep_fifteen_then_three() -> None:
    candidates = [
        {"index": idx, "title": f"Paper {idx}", "abstract": f"Abstract {idx}", "label": "gold" if idx < 2 else "negative_t1"}
        for idx in range(75)
    ]
    record = {
        "sample_id": "sample/1",
        "research_question": "question",
        "background_survey": "survey",
        "candidates": candidates,
    }
    pred = run_retrieve_record(record, MockClient())

    assert len(pred["selected_round1_titles"]) == 15
    assert len(pred["selected_round2_titles"]) == 3
    assert pred["selected_round1_titles"][:3] == ["Paper 0", "Paper 1", "Paper 2"]
    assert pred["selected_round2_titles"] == ["Paper 0", "Paper 1", "Paper 2"]

    score = score_retrieve([pred], [record])
    assert score["summary"]["round1"]["gold"] == 1.0
    assert score["summary"]["round2"]["gold"] == 1.0
