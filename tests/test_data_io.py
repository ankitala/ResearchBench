from researchbench.data_io import (
    best_title_match,
    contains_forbidden_text,
    normalize_doi,
    normalize_title,
)


def test_normalize_doi_removes_doi_punctuation() -> None:
    assert normalize_doi("10.1016/j.ascom.2023.100771") == "101016jascom2023100771"
    assert normalize_doi("10.1038-S41586-024-07860-9") == "101038s41586024078609"


def test_normalize_title_and_best_title_match() -> None:
    candidates = [
        {"title": "Dictionary learning allows model-free pseudotime estimation of transcriptomic data"},
        {"title": "Perfect metamaterial absorber"},
    ]
    idx, score = best_title_match(
        "Dictionary Learning Allows Model Free Pseudotime Estimation of Transcriptomic Data",
        candidates,
    )
    assert idx == 0
    assert score == 1.0
    assert normalize_title("PtTe2-based type-II dirac semimetal") == "ptte2 based type ii dirac semimetal"


def test_forbidden_text_detection_avoids_common_scientific_false_positives() -> None:
    row = {
        "url": "https://github.com/example/project",
        "title": "Dusk-to-nighttime enhancement of mid-latitude Nm F2",
    }
    assert not contains_forbidden_text(row)
    assert contains_forbidden_text({"path": "/Users/alice/private/file.json"})
    assert contains_forbidden_text({"token": "sk-proj-" + "x" * 32})
