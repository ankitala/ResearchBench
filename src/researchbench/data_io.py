"""File, JSONL, and matching utilities."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


API_KEY_RE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b|api[_-]?key", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"((?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home|mnt)/)")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_doi(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def normalize_title(value: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def title_word_set(value: str | None) -> set[str]:
    return set(normalize_title(value).split())


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_title_match(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    threshold: float = 0.72,
) -> tuple[int | None, float]:
    """Return the best candidate index for a generated title."""

    wanted = normalize_title(title)
    if not wanted:
        return None, 0.0
    for idx, candidate in enumerate(candidates):
        if normalize_title(candidate.get("title", "")) == wanted:
            return idx, 1.0
    wanted_words = title_word_set(title)
    best_idx: int | None = None
    best_score = 0.0
    for idx, candidate in enumerate(candidates):
        score = jaccard_similarity(wanted_words, title_word_set(candidate.get("title", "")))
        if score > best_score:
            best_idx, best_score = idx, score
    if best_score >= threshold:
        return best_idx, best_score
    return None, best_score


def contains_forbidden_text(value: Any) -> bool:
    """Detect accidental leakage of local paths or API keys in structured rows."""

    if isinstance(value, str):
        return bool(API_KEY_RE.search(value) or LOCAL_PATH_RE.search(value))
    if isinstance(value, dict):
        return any(contains_forbidden_text(key) or contains_forbidden_text(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        return any(contains_forbidden_text(item) for item in value)
    return False


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def paper_pair(item: Any) -> tuple[str, str] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return clean_text(item[0]), clean_text(item[1])
    if isinstance(item, dict):
        title = clean_text(item.get("title"))
        abstract = clean_text(item.get("abstract"))
        if title or abstract:
            return title, abstract
    return None
