import json
from pathlib import Path

from researchbench.builders import _label_candidates


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_label_candidates_maps_d0_to_d3_and_preserves_unknown(tmp_path: Path) -> None:
    _write_json(tmp_path / "d0.json", [["Gold Paper", "gold abstract"]])
    _write_json(tmp_path / "d1.json", [["Tier One Negative", "t1 abstract"]])
    _write_json(tmp_path / "d2.json", [["Tier Two Negative", "t2 abstract"]])
    _write_json(tmp_path / "d3.json", [["Tier Three Negative", "t3 abstract"]])

    candidates = [
        {"title": "Gold Paper", "abstract": "gold abstract"},
        {"title": "Tier One Negative", "abstract": "t1 abstract"},
        {"title": "Tier Two Negative", "abstract": "t2 abstract"},
        {"title": "Tier Three Negative", "abstract": "t3 abstract"},
        {"title": "Unmatched Candidate", "abstract": "unknown abstract"},
    ]
    labeled, issues = _label_candidates(tmp_path, candidates)

    assert [row["label"] for row in labeled] == [
        "gold",
        "negative_t1",
        "negative_t2",
        "negative_t3",
        "negative_unknown",
    ]
    assert issues == [
        {
            "candidate_index": 4,
            "title": "Unmatched Candidate",
            "best_match_score": "0.000",
        }
    ]
