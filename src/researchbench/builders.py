"""Build standardized ResearchBench JSONL data from the original materials."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import (
    best_title_match,
    clean_text,
    contains_forbidden_text,
    normalize_doi,
    paper_pair,
    read_json,
    write_csv,
    write_json,
    write_jsonl,
)
from .schemas import DEFAULT_GENERATION_PARAMS, DEFAULT_RETRIEVE_PARAMS, LABEL_BY_FILE, LABEL_PRIORITY, SUBJECTS


def _partial_python_root(source_root: Path) -> Path:
    return source_root / "researchbench的部分源代码" / "python"


def _distance_root(source_root: Path) -> Path:
    return _partial_python_root(source_root) / "spider" / "distance"


def _ranking_roots(source_root: Path) -> list[Path]:
    return [
        source_root / "researchbench" / "python" / "ranking_fan" / "ranking",
        _partial_python_root(source_root) / "ranking_fan" / "ranking",
    ]


def _sample_id(discipline: str, doi: str) -> str:
    return f"{discipline}/{normalize_doi(doi)}"


def load_papers(source_root: str | Path) -> list[dict[str, Any]]:
    source_root = Path(source_root)
    rows: list[dict[str, Any]] = []
    for discipline in SUBJECTS:
        path = _partial_python_root(source_root) / discipline / "result.json"
        data = read_json(path)
        for idx, item in enumerate(data):
            if clean_text(item.get("sufficiency_tag")).lower() != "yes":
                continue
            doi = clean_text(item.get("doi"))
            inspirations = []
            for insp in item.get("inspiration", []) or []:
                inspirations.append(
                    {
                        "title": clean_text(insp.get("title")),
                        "abstract": clean_text(insp.get("abstract")),
                        "relation": clean_text(insp.get("relation")),
                        "insp": clean_text(insp.get("insp")),
                    }
                )
            rows.append(
                {
                    "sample_id": _sample_id(discipline, doi),
                    "discipline": discipline,
                    "doi": doi,
                    "title": clean_text(item.get("title")),
                    "abstract": clean_text(item.get("abstract")),
                    "research_question": clean_text(item.get("research_question")),
                    "background_survey": clean_text(item.get("background_little_survey")),
                    "main_hypothesis": clean_text(item.get("main_hypothesis")),
                    "fine_grained_hypothesis": clean_text(item.get("fine_grained_hypothesis")),
                    "experiments_details": clean_text(item.get("experiments_details")),
                    "inspirations": inspirations,
                    "source_index": idx,
                }
            )
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def _folder_map(root: Path) -> dict[tuple[str, str], Path]:
    mapping: dict[tuple[str, str], Path] = {}
    for discipline in SUBJECTS:
        discipline_root = root / discipline
        if not discipline_root.exists():
            continue
        for folder in discipline_root.iterdir():
            if folder.is_dir():
                mapping[(discipline, normalize_doi(folder.name))] = folder
    return mapping


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    pairs: list[dict[str, Any]] = []
    for item in read_json(path):
        parsed = paper_pair(item)
        if parsed is None:
            continue
        title, abstract = parsed
        pairs.append({"title": title, "abstract": abstract})
    return pairs


def _label_candidates(folder: Path, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_title_maps: dict[str, list[dict[str, Any]]] = {}
    for file_name, label in LABEL_BY_FILE.items():
        label_title_maps[label] = _load_pairs(folder / file_name)

    labeled = []
    issues: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        label = "negative_unknown"
        matched_file = ""
        match_score = 0.0
        for candidate_label in LABEL_PRIORITY:
            label_candidates = label_title_maps.get(candidate_label, [])
            match_idx, score = best_title_match(candidate["title"], label_candidates, threshold=0.72)
            if match_idx is not None:
                label = candidate_label
                matched_file = next(k for k, v in LABEL_BY_FILE.items() if v == candidate_label)
                match_score = score
                break
            match_score = max(match_score, score)
        row = {
            "index": idx,
            "title": candidate["title"],
            "abstract": candidate["abstract"],
            "label": label,
        }
        if matched_file:
            row["source_file"] = matched_file
        labeled.append(row)
        if label == "negative_unknown":
            issues.append(
                {
                    "candidate_index": idx,
                    "title": candidate["title"],
                    "best_match_score": f"{match_score:.3f}",
                }
            )
    return labeled, issues


def build_retrieve_and_generation(
    papers: list[dict[str, Any]],
    source_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(source_root)
    folders = _folder_map(_distance_root(source_root))
    retrieve_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for paper in papers:
        key = (paper["discipline"], normalize_doi(paper["doi"]))
        folder = folders.get(key)
        if folder is None:
            issues.append({"sample_id": paper["sample_id"], "issue": "missing_distance_folder"})
            continue
        merge = _load_pairs(folder / "merge.json")
        if len(merge) != 75:
            issues.append(
                {
                    "sample_id": paper["sample_id"],
                    "issue": "candidate_count_not_75",
                    "value": len(merge),
                }
            )
        candidates, candidate_issues = _label_candidates(folder, merge)
        label_counts = Counter(candidate["label"] for candidate in candidates)
        for issue in candidate_issues:
            issues.append({"sample_id": paper["sample_id"], "issue": "candidate_unknown_label", **issue})
        gold = [candidate for candidate in candidates if candidate["label"] == "gold"]
        if not gold:
            issues.append({"sample_id": paper["sample_id"], "issue": "no_gold_in_candidates"})
        retrieve_rows.append(
            {
                "sample_id": paper["sample_id"],
                "discipline": paper["discipline"],
                "doi": paper["doi"],
                "doi_folder": folder.name,
                "research_question": paper["research_question"],
                "background_survey": paper["background_survey"],
                "candidates": candidates,
                "gold_titles": [candidate["title"] for candidate in gold],
                "label_counts": dict(label_counts),
                "default_params": dict(DEFAULT_RETRIEVE_PARAMS),
            }
        )
        generation_rows.append(
            {
                "sample_id": paper["sample_id"],
                "discipline": paper["discipline"],
                "doi": paper["doi"],
                "research_question": paper["research_question"],
                "background_survey": paper["background_survey"],
                "gold_inspirations": [
                    {"title": candidate["title"], "abstract": candidate["abstract"]} for candidate in gold
                ],
                "true_retrieve": [
                    {"title": candidate["title"], "abstract": candidate["abstract"]} for candidate in gold
                ],
                "gold_hypothesis": paper["main_hypothesis"],
                "fine_grained_hypothesis": paper["fine_grained_hypothesis"],
                "experiments_details": paper["experiments_details"],
                "default_params": dict(DEFAULT_GENERATION_PARAMS),
            }
        )
    return retrieve_rows, generation_rows, issues


def build_ranking(
    papers: list[dict[str, Any]],
    source_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(source_root)
    roots = [root for root in _ranking_roots(source_root) if root.exists()]
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def find_ranking_file(paper: dict[str, Any]) -> Path | None:
        for root in roots:
            discipline_root = root / paper["discipline"]
            if not discipline_root.exists():
                continue
            for prefix in ("random_", ""):
                expected = discipline_root / f"{prefix}{paper['doi'].replace('/', '-')}.json"
                if expected.exists():
                    return expected
            candidates = []
            for path in discipline_root.glob("*.json"):
                stem = path.stem.removeprefix("random_")
                if normalize_doi(stem) == normalize_doi(paper["doi"]):
                    candidates.append(path)
            if candidates:
                candidates.sort(key=lambda path: (0 if path.name.startswith("random_") else 1, path.name))
                return candidates[0]
        return None

    for paper in papers:
        path = find_ranking_file(paper)
        if path is None:
            issues.append({"sample_id": paper["sample_id"], "issue": "missing_ranking_task"})
            continue
        data = read_json(path)
        fake = [clean_text(x) for x in data.get("fake generate hypothesis", []) if clean_text(x)]
        model = [clean_text(x) for x in data.get("model generate hypothesis", []) if clean_text(x)]
        if len(fake) < 5 or len(model) < 10:
            issues.append(
                {
                    "sample_id": paper["sample_id"],
                    "issue": "ranking_negative_count_too_small",
                    "fake_count": len(fake),
                    "model_count": len(model),
                }
            )
            continue
        rows.append(
            {
                "sample_id": paper["sample_id"],
                "discipline": paper["discipline"],
                "doi": paper["doi"],
                "research_question": clean_text(data.get("Background Question")) or paper["research_question"],
                "gold_hypothesis": clean_text(data.get("Main hypothesis")) or paper["main_hypothesis"],
                "fake_negative_hypotheses": fake[:5],
                "model_negative_hypotheses": model[:10],
                "source_variant": "random" if path.name.startswith("random_") else "direct",
            }
        )
    rows.sort(key=lambda row: row["sample_id"])
    return rows, issues


def _tiny_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    by_subject: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_subject.setdefault(row["discipline"], row)
    selected = list(by_subject.values())
    if len(selected) >= limit:
        return selected[:limit]
    selected_ids = {row["sample_id"] for row in selected}
    for row in rows:
        if row["sample_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["sample_id"])
        if len(selected) >= limit:
            break
    return selected


def build_all(source_root: str | Path, out: str | Path) -> dict[str, int]:
    out = Path(out)
    source_root = Path(source_root)
    papers = load_papers(source_root)
    retrieve, generation, retrieve_issues = build_retrieve_and_generation(papers, source_root)
    ranking, ranking_issues = build_ranking(papers, source_root)

    write_jsonl(out / "papers.jsonl", papers)
    write_jsonl(out / "tasks" / "retrieve.jsonl", retrieve)
    write_jsonl(out / "tasks" / "generation.jsonl", generation)
    write_jsonl(out / "tasks" / "ranking.jsonl", ranking)

    write_jsonl(out / "tiny" / "papers.jsonl", _tiny_rows(papers))
    write_jsonl(out / "tiny" / "retrieve.jsonl", _tiny_rows(retrieve))
    write_jsonl(out / "tiny" / "generation.jsonl", _tiny_rows(generation))
    write_jsonl(out / "tiny" / "ranking.jsonl", _tiny_rows(ranking))

    write_csv(
        out / "validation" / "retrieve_build_issues.csv",
        retrieve_issues,
        ["sample_id", "issue", "candidate_index", "title", "best_match_score", "value"],
    )
    write_csv(
        out / "validation" / "ranking_build_issues.csv",
        ranking_issues,
        ["sample_id", "issue", "fake_count", "model_count"],
    )
    summary = {
        "papers": len(papers),
        "retrieve": len(retrieve),
        "generation": len(generation),
        "ranking": len(ranking),
        "retrieve_issues": len(retrieve_issues),
        "ranking_issues": len(ranking_issues),
    }
    write_json(out / "validation" / "build_summary.json", summary)
    return summary


def validate_data_dir(data_dir: str | Path) -> tuple[bool, list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    issues: list[dict[str, Any]] = []

    def add(file_name: str, sample_id: str, issue: str, value: Any = "") -> None:
        issues.append({"file": file_name, "sample_id": sample_id, "issue": issue, "value": value})

    from .data_io import iter_jsonl

    for relative in [
        "papers.jsonl",
        "tasks/retrieve.jsonl",
        "tasks/generation.jsonl",
        "tasks/ranking.jsonl",
    ]:
        path = data_dir / relative
        if not path.exists():
            add(relative, "", "missing_file")
            continue
        seen: set[str] = set()
        for row in iter_jsonl(path):
            sample_id = clean_text(row.get("sample_id"))
            if not sample_id:
                add(relative, "", "missing_sample_id")
            elif sample_id in seen:
                add(relative, sample_id, "duplicate_sample_id")
            seen.add(sample_id)
            if contains_forbidden_text(row):
                add(relative, sample_id, "forbidden_local_path_or_api_key")
            if relative == "papers.jsonl":
                for key in ["research_question", "main_hypothesis", "doi", "discipline"]:
                    if not clean_text(row.get(key)):
                        add(relative, sample_id, f"empty_{key}")
            elif relative == "tasks/retrieve.jsonl":
                candidates = row.get("candidates", [])
                if len(candidates) != 75:
                    add(relative, sample_id, "candidate_count_not_75", len(candidates))
                if not row.get("gold_titles"):
                    add(relative, sample_id, "empty_gold_titles")
                if not any(candidate.get("label") == "gold" for candidate in candidates):
                    add(relative, sample_id, "no_gold_candidate")
                if not clean_text(row.get("research_question")):
                    add(relative, sample_id, "empty_research_question")
            elif relative == "tasks/generation.jsonl":
                if not row.get("gold_inspirations"):
                    add(relative, sample_id, "empty_gold_inspirations")
                if not clean_text(row.get("gold_hypothesis")):
                    add(relative, sample_id, "empty_gold_hypothesis")
            elif relative == "tasks/ranking.jsonl":
                if len(row.get("fake_negative_hypotheses", [])) != 5:
                    add(relative, sample_id, "fake_negative_count_not_5")
                if len(row.get("model_negative_hypotheses", [])) != 10:
                    add(relative, sample_id, "model_negative_count_not_10")
    write_csv(
        data_dir / "validation" / "validation_issues.csv",
        issues,
        ["file", "sample_id", "issue", "value"],
    )
    return not issues, issues
