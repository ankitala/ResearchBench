"""Run and score hypothesis composition with MOOSE-Chem-compatible prompts."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any

from .clients import ModelClient
from .data_io import best_title_match
from .prompts import (
    background_survey_text,
    generation_additional_inspiration_screening_prompt,
    generation_between_inspiration_recombine_prompt,
    generation_between_inspiration_recombine_refine_prompt,
    generation_distinct_mutation_prompt,
    generation_feedback_prompt,
    generation_initial_prompt,
    generation_refine_prompt,
    generation_same_inspiration_recombine_prompt,
    generation_same_inspiration_recombine_refine_prompt,
    generation_score_prompt,
    generation_self_eval_prompt,
)
from .retrieve import parse_retrieve_response
from .schemas import DEFAULT_GENERATION_PARAMS


def _strip_reasoning_model_content(text: str) -> str:
    answer = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer:
        return answer.group(1).strip()
    after_think = re.search(r"</think>\s*\n(.+)", text, re.IGNORECASE | re.DOTALL)
    if after_think:
        return after_think.group(1).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^.*</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip() or text


def extract_marker_field(text: str, field_name: str) -> str | None:
    plain = re.sub(r"[\*_]+", "", _strip_reasoning_model_content(text))
    pattern = rf"{re.escape(field_name)}\s*starts\s*:?\s*([\s\S]+?)\s*{re.escape(field_name)}\s*ends"
    match = re.search(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
    if match:
        value = match.group(1).strip()
        return value or None
    pattern = rf"{re.escape(field_name)}\s*[:：]\s*(.+?)(?:\n|$)"
    match = re.search(pattern, plain, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip().strip("*").strip()
        return value or None
    return None


def parse_hypothesis_response(text: str) -> tuple[str, str]:
    hypothesis = (
        extract_marker_field(text, "Hypothesis")
        or extract_marker_field(text, "Refined Hypothesis")
        or ""
    )
    reasoning = extract_marker_field(text, "Key Reasoning") or ""
    if not hypothesis:
        legacy = re.search(
            r"Hypothesis\s*[:：]\s*(?P<hyp>.*?)(?:\n\s*Reasoning Process\s*[:：]\s*(?P<reason>.*))?\Z",
            text.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if legacy:
            hypothesis = legacy.group("hyp").strip()
            reasoning = (legacy.group("reason") or "").strip()
        else:
            hypothesis = text.strip()
    return hypothesis, reasoning


def parse_matched_score(text: str) -> float | None:
    marked = extract_marker_field(text, "Matched score")
    if marked is not None:
        numbers = re.findall(r"[0-5](?:\.\d+)?", marked)
        if numbers:
            return float(numbers[0])
    match = re.search(r"Matched\s*score\s*[:：]\s*(?P<score>[0-5](?:\.\d+)?)", text, re.IGNORECASE)
    if not match:
        return None
    score = float(match.group("score"))
    return score if 0 <= score <= 5 else None


def parse_self_eval_scores(text: str) -> tuple[list[int], list[str]]:
    score_types = ["Validness", "Novelty", "Significance", "Specificity"]
    scores: list[int] = []
    reasons: list[str] = []
    for score_type in score_types:
        reason = extract_marker_field(text, f"{score_type} Reason") or ""
        score_text = extract_marker_field(text, f"{score_type} Score")
        score: int | None = None
        if score_text:
            numbers = re.findall(r"\d+", score_text)
            if numbers:
                score = int(numbers[0])
        if score is None:
            match = re.search(rf"{score_type}\s+(?:score\s*)?[:：]\s*(\d+)", text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
        if score is None or not 1 <= score <= 5:
            score = 3
        scores.append(score)
        reasons.append(reason)
    return scores, reasons


def _with_default_reason(inspirations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for inspiration in inspirations:
        normalized.append(
            {
                "title": inspiration.get("title", ""),
                "abstract": inspiration.get("abstract", ""),
                "reason": inspiration.get("reason", inspiration.get("insp", "Not provided yet.")),
            }
        )
    return normalized


def _generation_record_view(record: dict[str, Any], *, use_background_survey: bool) -> dict[str, Any]:
    return {
        **record,
        "background_survey": background_survey_text(
            record.get("background_survey", ""),
            use_background_survey=use_background_survey,
        ),
    }


def _self_evaluate_hypothesis(hypothesis: str, client: ModelClient) -> dict[str, Any]:
    response = client.generate(generation_self_eval_prompt(hypothesis), temperature=1.0)
    scores, reasons = parse_self_eval_scores(response)
    return {
        "scores": scores,
        "score_reasons": reasons,
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "raw_response": response,
    }


def estimate_generation_model_calls(
    record: dict[str, Any],
    *,
    num_mutations: int = int(DEFAULT_GENERATION_PARAMS["num_mutations"]),
    num_itr_self_refine: int = int(DEFAULT_GENERATION_PARAMS["num_itr_self_refine"]),
    max_inspiration_search_steps: int = int(DEFAULT_GENERATION_PARAMS["max_inspiration_search_steps"]),
    num_screening_window_size: int = int(DEFAULT_GENERATION_PARAMS["num_screening_window_size"]),
    num_screening_keep_size: int = int(DEFAULT_GENERATION_PARAMS["num_screening_keep_size"]),
    recom_num_beam_size: int = int(DEFAULT_GENERATION_PARAMS["recom_num_beam_size"]),
) -> int:
    inspirations = list(
        record.get("gold_inspirations")
        or record.get("true_retrieve")
        or record.get("ture_retrieve")
        or []
    )[:max_inspiration_search_steps]
    num_inspirations = len(inspirations)
    if num_inspirations == 0:
        return 0
    calls_per_hypothesis_node = 2 * num_itr_self_refine + 1
    first_step_nodes_per_inspiration = num_mutations + (1 if num_mutations > 1 else 0)
    total = num_inspirations * first_step_nodes_per_inspiration * calls_per_hypothesis_node

    ranked_nodes = [{"used": {idx}} for idx in range(num_inspirations)]
    for _step_id in range(2, min(max_inspiration_search_steps, num_inspirations) + 1):
        ranked_nodes = ranked_nodes[: max(1, min(recom_num_beam_size, len(ranked_nodes)))]
        next_nodes = []
        for ranked_node in ranked_nodes:
            other_count = num_inspirations - len(ranked_node["used"])
            if other_count <= 0:
                continue
            if other_count > num_screening_keep_size:
                for start in range(0, other_count, num_screening_window_size):
                    window_count = min(num_screening_window_size, other_count - start)
                    if window_count > num_screening_keep_size:
                        total += 1
            for other_idx in range(num_inspirations):
                if other_idx in ranked_node["used"]:
                    continue
                total += calls_per_hypothesis_node
                next_nodes.append({"used": ranked_node["used"] | {other_idx}})
        if not next_nodes:
            break
        ranked_nodes = next_nodes
    return total


def _generate_with_refinement(
    record: dict[str, Any],
    inspiration: dict[str, Any],
    client: ModelClient,
    *,
    num_itr_self_refine: int,
    other_mutations: list[str] | None = None,
    recombination_type: int = 0,
    core_hypothesis: str | None = None,
    other_inspiration: dict[str, Any] | None = None,
    other_hypothesis: str | None = None,
) -> dict[str, Any]:
    trace = []
    previous_hypothesis: str | None = None
    feedback: str | None = None
    current_hypothesis = ""
    current_reasoning = ""
    for refine_iter in range(num_itr_self_refine):
        if refine_iter == 0:
            if recombination_type == 2:
                assert core_hypothesis is not None and other_inspiration is not None and other_hypothesis is not None
                prompt = generation_between_inspiration_recombine_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    core_hypothesis,
                    other_inspiration,
                    other_hypothesis,
                )
            elif recombination_type == 1:
                prompt = generation_same_inspiration_recombine_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    other_mutations or [],
                )
            elif other_mutations:
                prompt = generation_distinct_mutation_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    other_mutations,
                )
            else:
                prompt = generation_initial_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                )
        else:
            assert previous_hypothesis is not None and feedback is not None
            if recombination_type == 2:
                assert core_hypothesis is not None and other_inspiration is not None and other_hypothesis is not None
                prompt = generation_between_inspiration_recombine_refine_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    core_hypothesis,
                    other_inspiration,
                    other_hypothesis,
                    previous_hypothesis,
                    feedback,
                )
            elif recombination_type == 1:
                prompt = generation_same_inspiration_recombine_refine_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    other_mutations or [],
                    previous_hypothesis,
                    feedback,
                )
            else:
                prompt = generation_refine_prompt(
                    record["research_question"],
                    record.get("background_survey", ""),
                    inspiration,
                    previous_hypothesis,
                    feedback,
                )

        response = client.generate(prompt, temperature=1.0)
        current_hypothesis, current_reasoning = parse_hypothesis_response(response)
        feedback_prompt = generation_feedback_prompt(current_hypothesis, current_reasoning)
        feedback = client.generate(feedback_prompt, temperature=1.0)
        trace.append(
            {
                "iteration": refine_iter,
                "prompt": prompt,
                "raw_response": response,
                "hypothesis": current_hypothesis,
                "reasoning": current_reasoning,
                "feedback": feedback,
            }
        )
        previous_hypothesis = current_hypothesis

    self_eval = _self_evaluate_hypothesis(current_hypothesis, client) if current_hypothesis else {
        "scores": [3, 3, 3, 3],
        "score_reasons": ["", "", "", ""],
        "average_score": 3.0,
        "raw_response": "",
    }
    return {
        "hypothesis": current_hypothesis,
        "reasoning": current_reasoning,
        "feedback": feedback or "",
        "self_eval": self_eval,
        "trace": trace,
    }


def _generate_for_one_inspiration(
    record: dict[str, Any],
    inspiration: dict[str, Any],
    client: ModelClient,
    *,
    num_mutations: int,
    num_itr_self_refine: int,
) -> dict[str, Any]:
    mutations: dict[str, Any] = {}
    first = _generate_with_refinement(
        record,
        inspiration,
        client,
        num_itr_self_refine=num_itr_self_refine,
    )
    mutations["0"] = first
    for mutation_id in range(1, num_mutations):
        other_mutations = [mutations[key]["hypothesis"] for key in mutations]
        mutations[str(mutation_id)] = _generate_with_refinement(
            record,
            inspiration,
            client,
            num_itr_self_refine=num_itr_self_refine,
            other_mutations=other_mutations,
        )
    if num_mutations > 1:
        other_mutations = [mutations[key]["hypothesis"] for key in mutations]
        mutations["recom"] = _generate_with_refinement(
            record,
            inspiration,
            client,
            num_itr_self_refine=num_itr_self_refine,
            other_mutations=other_mutations,
            recombination_type=1,
        )
    best_key, best_node = max(
        mutations.items(),
        key=lambda item: item[1]["self_eval"]["average_score"],
    )
    return {
        "inspiration_title": inspiration.get("title", ""),
        "mutations": mutations,
        "best_mutation_id": best_key,
        "best_hypothesis": best_node["hypothesis"],
        "best_self_eval": best_node["self_eval"],
    }


def _screen_complementary_inspirations(
    record: dict[str, Any],
    core_inspiration: dict[str, Any],
    core_hypothesis: str,
    other_mutations: list[dict[str, Any]],
    client: ModelClient,
    *,
    window_size: int,
    keep_size: int,
) -> list[dict[str, Any]]:
    if not other_mutations:
        return []
    if len(other_mutations) <= keep_size:
        return other_mutations
    selected: list[dict[str, Any]] = []
    for start in range(0, len(other_mutations), window_size):
        window = other_mutations[start : start + window_size]
        if len(window) <= keep_size:
            selected.extend(window)
            continue
        if client.is_mock:
            selected.extend(window[:keep_size])
            continue
        prompt = generation_additional_inspiration_screening_prompt(
            record["research_question"],
            record.get("background_survey", ""),
            core_inspiration,
            core_hypothesis,
            window,
            keep_size=keep_size,
        )
        response = client.generate(prompt, temperature=0.0)
        used: set[int] = set()
        for parsed in parse_retrieve_response(response):
            idx, score = best_title_match(parsed["title"], window, threshold=0.62)
            if idx is None or idx in used:
                continue
            used.add(idx)
            selected.append({**window[idx], "match_score": score})
            if len(used) >= keep_size:
                break
    return selected


def _inter_inspiration_recombine(
    record: dict[str, Any],
    inspiration_results: list[dict[str, Any]],
    inspirations: list[dict[str, Any]],
    client: ModelClient,
    *,
    num_itr_self_refine: int,
    max_steps: int,
    num_screening_window_size: int,
    num_screening_keep_size: int,
    recom_num_beam_size: int,
    recombination_concurrency: int = 1,
    client_factory: Callable[[], ModelClient] | None = None,
) -> list[dict[str, Any]]:
    inter_results = []
    client_factory = client_factory or (lambda: client)
    ranked_nodes: list[dict[str, Any]] = [
        {
            "inspiration_index": idx,
            "hypothesis": item["best_hypothesis"],
            "self_eval": item["best_self_eval"],
            "mutation_trail": [item["best_mutation_id"]],
        }
        for idx, item in enumerate(inspiration_results)
    ]
    for step_id in range(2, min(max_steps, len(inspirations)) + 1):
        ranked_nodes = sorted(
            ranked_nodes,
            key=lambda item: item["self_eval"]["average_score"],
            reverse=True,
        )
        if not ranked_nodes:
            break
        beam_size = min(recom_num_beam_size, len(ranked_nodes))
        score_threshold = ranked_nodes[beam_size - 1]["self_eval"]["average_score"]
        filtered_nodes = [
            item for item in ranked_nodes if item["self_eval"]["average_score"] >= score_threshold
        ]
        best_first_step_by_inspiration = {
            idx: {
                "title": inspirations[idx].get("title", ""),
                "abstract": inspirations[idx].get("abstract", ""),
                "hypothesis": result["best_hypothesis"],
            }
            for idx, result in enumerate(inspiration_results)
        }
        generation_tasks: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], int]] = []
        for node_id, ranked_node in enumerate(filtered_nodes):
            core_idx = ranked_node["inspiration_index"]
            core_inspiration = inspirations[core_idx]
            used_titles = {core_inspiration.get("title", "")}
            for trail_item in ranked_node["mutation_trail"]:
                used_titles.update(part for part in str(trail_item).split(";") if part)
            other_mutations = [
                mutation
                for idx, mutation in best_first_step_by_inspiration.items()
                if idx != core_idx and mutation["title"] not in used_titles
            ]
            selected_other_mutations = _screen_complementary_inspirations(
                record,
                core_inspiration,
                ranked_node["hypothesis"],
                other_mutations,
                client_factory(),
                window_size=num_screening_window_size,
                keep_size=num_screening_keep_size,
            )
            for other in selected_other_mutations:
                other_idx = next(
                    (
                        idx
                        for idx, inspiration in enumerate(inspirations)
                        if inspiration.get("title", "") == other.get("title", "")
                    ),
                    None,
                )
                if other_idx is None:
                    continue
                generation_tasks.append((node_id, ranked_node, core_inspiration, other, other_idx))
        def run_task(task: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], int]) -> dict[str, Any]:
            node_id, ranked_node, core_inspiration, other, other_idx = task
            generated = _generate_with_refinement(
                record,
                core_inspiration,
                client_factory(),
                num_itr_self_refine=num_itr_self_refine,
                recombination_type=2,
                core_hypothesis=ranked_node["hypothesis"],
                other_inspiration=inspirations[other_idx],
                other_hypothesis=other["hypothesis"],
            )
            trail = ranked_node["mutation_trail"] + [other["title"], f"inter_recom_{step_id - 1}"]
            return {
                "step": step_id,
                "node_id": node_id,
                "core_inspiration_title": core_inspiration.get("title", ""),
                "other_inspiration_title": other.get("title", ""),
                "hypothesis": generated["hypothesis"],
                "self_eval": generated["self_eval"],
                "mutation_trail": trail,
                "node": generated,
                "inspiration_index": ranked_node["inspiration_index"],
            }

        if recombination_concurrency > 1 and len(generation_tasks) > 1:
            with ThreadPoolExecutor(max_workers=recombination_concurrency) as executor:
                step_results = list(executor.map(run_task, generation_tasks))
        else:
            step_results = [run_task(task) for task in generation_tasks]
        next_ranked_nodes: list[dict[str, Any]] = []
        for result in step_results:
            inter_results.append({key: value for key, value in result.items() if key != "inspiration_index"})
            next_ranked_nodes.append(
                {
                    "inspiration_index": result["inspiration_index"],
                    "hypothesis": result["hypothesis"],
                    "self_eval": result["self_eval"],
                    "mutation_trail": result["mutation_trail"],
                }
            )
        if not next_ranked_nodes:
            break
        ranked_nodes = next_ranked_nodes
    return inter_results


def run_generation_record(
    record: dict[str, Any],
    client: ModelClient,
    *,
    num_mutations: int = int(DEFAULT_GENERATION_PARAMS["num_mutations"]),
    num_itr_self_refine: int = int(DEFAULT_GENERATION_PARAMS["num_itr_self_refine"]),
    max_inspiration_search_steps: int = int(DEFAULT_GENERATION_PARAMS["max_inspiration_search_steps"]),
    use_background_survey: bool = bool(DEFAULT_GENERATION_PARAMS["use_background_survey"]),
    num_screening_window_size: int = int(DEFAULT_GENERATION_PARAMS["num_screening_window_size"]),
    num_screening_keep_size: int = int(DEFAULT_GENERATION_PARAMS["num_screening_keep_size"]),
    recom_num_beam_size: int = int(DEFAULT_GENERATION_PARAMS["recom_num_beam_size"]),
    inner_concurrency: int = 1,
    client_factory: Callable[[], ModelClient] | None = None,
) -> dict[str, Any]:
    inner_concurrency = max(1, inner_concurrency)
    client_factory = client_factory or (lambda: client)
    record_for_prompt = _generation_record_view(record, use_background_survey=use_background_survey)
    inspirations = _with_default_reason(
        list(
            record.get("gold_inspirations")
            or record.get("true_retrieve")
            or record.get("ture_retrieve")
            or []
        )[:max_inspiration_search_steps]
    )
    def run_inspiration(inspiration: dict[str, Any]) -> dict[str, Any]:
        return _generate_for_one_inspiration(
            record_for_prompt,
            inspiration,
            client_factory(),
            num_mutations=num_mutations,
            num_itr_self_refine=num_itr_self_refine,
        )

    if inner_concurrency > 1 and len(inspirations) > 1:
        with ThreadPoolExecutor(max_workers=inner_concurrency) as executor:
            inspiration_results = list(executor.map(run_inspiration, inspirations))
    else:
        inspiration_results = [run_inspiration(inspiration) for inspiration in inspirations]
    inter_results = _inter_inspiration_recombine(
        record_for_prompt,
        inspiration_results,
        inspirations,
        client_factory(),
        num_itr_self_refine=num_itr_self_refine,
        max_steps=max_inspiration_search_steps,
        num_screening_window_size=num_screening_window_size,
        num_screening_keep_size=num_screening_keep_size,
        recom_num_beam_size=recom_num_beam_size,
        recombination_concurrency=inner_concurrency,
        client_factory=client_factory,
    )
    candidates = [
        {
            "source": f"inspiration:{item['inspiration_title']}",
            "hypothesis": item["best_hypothesis"],
            "self_eval": item["best_self_eval"],
        }
        for item in inspiration_results
    ] + [
        {
            "source": f"inter_recom_step:{item['step']}",
            "hypothesis": item["hypothesis"],
            "self_eval": item["self_eval"],
        }
        for item in inter_results
    ]
    best_candidate = max(candidates, key=lambda item: item["self_eval"]["average_score"]) if candidates else None
    final_hypothesis = best_candidate["hypothesis"] if best_candidate else ""
    return {
        "sample_id": record["sample_id"],
        "generated_hypotheses": candidates,
        "final_hypothesis": final_hypothesis,
        "final_reasoning": best_candidate["source"] if best_candidate else "",
        "moosechem_trace": {
            "inspiration_results": inspiration_results,
            "inter_inspiration_recombinations": inter_results,
            "best_candidate": best_candidate,
            "params": {
                "if_use_gdth_insp": True,
                "num_mutations": num_mutations,
                "num_itr_self_refine": num_itr_self_refine,
                "max_inspiration_search_steps": max_inspiration_search_steps,
                "if_use_background_survey": use_background_survey,
                "num_screening_window_size": num_screening_window_size,
                "num_screening_keep_size": num_screening_keep_size,
                "recom_num_beam_size": recom_num_beam_size,
                "inner_concurrency": inner_concurrency,
                "if_mutate_inside_same_bkg_insp": True,
                "if_mutate_between_diff_insp": True,
                "if_self_explore": False,
            },
        },
    }


def score_generation(
    predictions: list[dict[str, Any]],
    data: list[dict[str, Any]],
    judge_client: ModelClient,
) -> dict[str, Any]:
    data_by_id = {row["sample_id"]: row for row in data}
    per_sample = []
    scores = []
    for pred in predictions:
        record = data_by_id.get(pred.get("sample_id"))
        if not record:
            continue
        score = pred.get("matched_score")
        reason = pred.get("matched_score_reason", "")
        if score is None:
            key_points = record.get("gold_key_points") or record["gold_hypothesis"]
            response = judge_client.generate(
                generation_score_prompt(pred.get("final_hypothesis", ""), record["gold_hypothesis"], key_points),
                temperature=0.0,
            )
            score = parse_matched_score(response)
            reason = response
        if score is None:
            continue
        score = float(score)
        scores.append(score)
        per_sample.append(
            {
                "sample_id": pred["sample_id"],
                "matched_score": score,
                "accuracy": score / 5.0,
                "reason": reason,
            }
        )
    average_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "summary": {
            "count": len(scores),
            "average_score": average_score,
            "accuracy": average_score / 5.0,
        },
        "per_sample": per_sample,
    }
