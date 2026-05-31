"""Run and score the inspiration retrieval task."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .clients import ModelClient
from .data_io import best_title_match
from .prompts import retrieve_prompt
from .schemas import DEFAULT_RETRIEVE_PARAMS


TITLE_REASON_RE = re.compile(
    r"Title\s*[:：]\s*(?P<title>.+?)(?:\n|$)(?:Reason\s*[:：]\s*(?P<reason>.*?)(?=\n\s*Title\s*[:：]|\Z))?",
    re.IGNORECASE | re.DOTALL,
)
TITLE_MARKER_RE = re.compile(
    r"(?:\*\*)?\s*Title\s+(?P<num>\d+)\s+starts\s*:\s*(?:\*\*)?\s*(?P<title>.+?)\s*(?:\*\*)?\s*Title\s+\1\s+ends\s*(?:\*\*)?",
    re.IGNORECASE | re.DOTALL,
)


def parse_retrieve_response(text: str) -> list[dict[str, str]]:
    results = []
    for match in TITLE_MARKER_RE.finditer(text.strip()):
        title = match.group("title").strip().strip("- ")
        if title:
            results.append({"title": title, "reason": ""})
    if results:
        return results
    for match in TITLE_REASON_RE.finditer(text.strip()):
        title = match.group("title").strip().strip("- ")
        reason = (match.group("reason") or "").strip()
        if title:
            results.append({"title": title, "reason": reason})
    return results


def _select_from_window(
    window: list[dict[str, Any]],
    record: dict[str, Any],
    client: ModelClient,
    *,
    keep_size: int,
    use_background_survey: bool,
    select_based_on_similarity: bool,
) -> list[dict[str, Any]]:
    if client.is_mock:
        return [
            {**candidate, "reason": "mock selection", "match_score": 1.0}
            for candidate in window[:keep_size]
        ]
    prompt = retrieve_prompt(
        record["research_question"],
        record.get("background_survey", ""),
        window,
        use_background_survey=use_background_survey,
        select_based_on_similarity=select_based_on_similarity,
    )
    response = client.generate(prompt, temperature=0.0)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for parsed in parse_retrieve_response(response):
        idx, score = best_title_match(parsed["title"], window, threshold=0.62)
        if idx is None or idx in used:
            continue
        used.add(idx)
        selected.append({**window[idx], "reason": parsed.get("reason", ""), "match_score": score})
        if len(selected) >= keep_size:
            break
    return selected


def run_retrieve_record(
    record: dict[str, Any],
    client: ModelClient,
    *,
    window_size: int = int(DEFAULT_RETRIEVE_PARAMS["window_size"]),
    keep_size: int = int(DEFAULT_RETRIEVE_PARAMS["keep_size"]),
    rounds: int = int(DEFAULT_RETRIEVE_PARAMS["rounds"]),
    use_background_survey: bool = bool(DEFAULT_RETRIEVE_PARAMS["use_background_survey"]),
    select_based_on_similarity: bool = bool(DEFAULT_RETRIEVE_PARAMS["select_based_on_similarity"]),
) -> dict[str, Any]:
    current = list(record["candidates"])
    round_outputs = []
    for round_id in range(1, rounds + 1):
        selected_this_round: list[dict[str, Any]] = []
        windows = [current[idx : idx + window_size] for idx in range(0, len(current), window_size)]
        for window_id, window in enumerate(windows):
            selected = _select_from_window(
                window,
                record,
                client,
                keep_size=keep_size,
                use_background_survey=use_background_survey,
                select_based_on_similarity=select_based_on_similarity,
            )
            selected_this_round.extend(selected)
            round_outputs.append(
                {
                    "round": round_id,
                    "window": window_id,
                    "selected": selected,
                }
            )
        current = selected_this_round
    return {
        "sample_id": record["sample_id"],
        "rounds": round_outputs,
        "selected_round1_titles": [
            item["title"]
            for output in round_outputs
            if output["round"] == 1
            for item in output["selected"]
        ],
        "selected_round2_titles": [
            item["title"]
            for output in round_outputs
            if output["round"] == 2
            for item in output["selected"]
        ],
    }


def score_retrieve(predictions: list[dict[str, Any]], data: list[dict[str, Any]]) -> dict[str, Any]:
    data_by_id = {row["sample_id"]: row for row in data}
    per_sample = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    hits: dict[str, Counter[str]] = defaultdict(Counter)

    for pred in predictions:
        record = data_by_id.get(pred.get("sample_id"))
        if not record:
            continue
        label_by_title = {candidate["title"]: candidate["label"] for candidate in record["candidates"]}
        label_counts = Counter(candidate["label"] for candidate in record["candidates"])
        sample_result: dict[str, Any] = {"sample_id": record["sample_id"], "rounds": {}}
        for round_name in ["selected_round1_titles", "selected_round2_titles"]:
            round_id = "round1" if round_name.endswith("1_titles") else "round2"
            selected_labels = Counter()
            for title in pred.get(round_name, []):
                idx, _ = best_title_match(title, record["candidates"], threshold=0.62)
                if idx is not None:
                    selected_labels[label_by_title[record["candidates"][idx]["title"]]] += 1
            round_result = {}
            for label, total in label_counts.items():
                totals[round_id][label] += total
                hits[round_id][label] += selected_labels[label]
                round_result[label] = selected_labels[label] / total if total else 0.0
            sample_result["rounds"][round_id] = round_result
        per_sample.append(sample_result)

    summary = {}
    for round_id in ["round1", "round2"]:
        summary[round_id] = {}
        for label, total in totals[round_id].items():
            summary[round_id][label] = hits[round_id][label] / total if total else 0.0
    return {"summary": summary, "per_sample": per_sample}
