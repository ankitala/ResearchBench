"""Shared constants and lightweight schema helpers for ResearchBench."""

from __future__ import annotations

SUBJECTS: tuple[str, ...] = (
    "Astronomy",
    "Biology",
    "Business",
    "Cell Biology",
    "Chemistry",
    "Earth Science",
    "Energy Science",
    "Environmental Science",
    "Law",
    "Material Science",
    "Math",
    "Physics",
)

LABEL_BY_FILE: dict[str, str] = {
    "d0.json": "gold",
    "d1.json": "negative_t1",
    "d2.json": "negative_t2",
    "d3.json": "negative_t3",
}

LABEL_PRIORITY: tuple[str, ...] = (
    "gold",
    "negative_t1",
    "negative_t2",
    "negative_t3",
)

DEFAULT_RETRIEVE_PARAMS: dict[str, int | bool] = {
    "window_size": 15,
    "keep_size": 3,
    "rounds": 2,
    "use_background_survey": False,
    "select_based_on_similarity": False,
}

DEFAULT_GENERATION_PARAMS: dict[str, int | bool] = {
    "if_use_gdth_insp": True,
    "num_mutations": 2,
    "num_itr_self_refine": 2,
    "max_inspiration_search_steps": 3,
    "use_background_survey": False,
    "num_screening_window_size": 12,
    "num_screening_keep_size": 3,
    "recom_num_beam_size": 15,
}

RANKING_ORDERS: tuple[str, ...] = ("res", "fan_1_res")
