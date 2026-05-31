"""Run and score hypothesis ranking."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .clients import ModelClient
from .prompts import ranking_prompt
from .schemas import RANKING_ORDERS


SELECTION_RE = re.compile(
    r"\*\*Selection\s+of\s+research\s+hypothesis\s+candidate\*\*\s*[:：]\s*candidate\s*(?P<num>[12])",
    re.IGNORECASE,
)


def parse_selection(text: str) -> int | None:
    match = SELECTION_RE.search(text)
    if match:
        return int(match.group("num"))
    loose = re.search(r"candidate\s*([12])", text, re.IGNORECASE)
    return int(loose.group(1)) if loose else None


def _compare(
    research_question: str,
    gold_hypothesis: str,
    negative_hypothesis: str,
    client: ModelClient,
    *,
    order: str,
) -> dict[str, Any]:
    if order == "res":
        candidate_1, candidate_2 = gold_hypothesis, negative_hypothesis
        negative_selection = 2
    elif order == "fan_1_res":
        candidate_1, candidate_2 = negative_hypothesis, gold_hypothesis
        negative_selection = 1
    else:
        raise ValueError(f"unsupported ranking order: {order}")
    prompt = ranking_prompt(research_question, candidate_1, candidate_2)
    response = client.generate(prompt, temperature=0.0)
    selection = parse_selection(response)
    if selection is None:
        selection = 1
    return {
        "selection": selection,
        "selected": "negative" if selection == negative_selection else "gold",
        "raw_response": response,
    }


def run_ranking_record(
    record: dict[str, Any],
    client: ModelClient,
    *,
    order: str = "both",
) -> dict[str, Any]:
    orders = list(RANKING_ORDERS) if order == "both" else [order]
    negatives = list(record.get("fake_negative_hypotheses", [])) + list(
        record.get("model_negative_hypotheses", [])
    )
    result: dict[str, Any] = {"sample_id": record["sample_id"], "orders": {}}
    for current_order in orders:
        rank_count = 16
        comparisons = []
        for idx, negative in enumerate(negatives):
            comparison = _compare(
                record["research_question"],
                record["gold_hypothesis"],
                negative,
                client,
                order=current_order,
            )
            if comparison["selected"] == "negative":
                rank_count -= 1
            comparisons.append({"negative_index": idx, **comparison})
        result["orders"][current_order] = {
            "rank": rank_count,
            "gold_wins": rank_count >= 9,
            "comparisons": comparisons,
        }
    return result


def score_ranking(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    order_counts = Counter()
    order_correct = Counter()
    sum_counts = Counter()
    pair_counts = Counter()
    pair_total = 0
    per_sample = []

    for pred in predictions:
        orders = pred.get("orders", {})
        sample_binary = {}
        for order_name in RANKING_ORDERS:
            if order_name not in orders:
                continue
            order_counts[order_name] += 1
            wins = bool(orders[order_name].get("gold_wins"))
            order_correct[order_name] += int(wins)
            sample_binary[order_name] = int(wins)
        if set(RANKING_ORDERS).issubset(sample_binary):
            sum_value = sample_binary["res"] + sample_binary["fan_1_res"]
            sum_counts[sum_value] += 1
            res_comps = orders["res"].get("comparisons", [])
            fan_comps = orders["fan_1_res"].get("comparisons", [])
            for res_item, fan_item in zip(res_comps, fan_comps):
                res_gold = res_item.get("selected") == "gold"
                fan_gold = fan_item.get("selected") == "gold"
                pair_total += 1
                if res_gold and fan_gold:
                    pair_counts["both_correct"] += 1
                elif not res_gold and not fan_gold:
                    pair_counts["both_wrong"] += 1
                else:
                    pair_counts["self_contradictory"] += 1
        per_sample.append({"sample_id": pred.get("sample_id"), "orders": sample_binary})

    order_accuracy = {
        order: order_correct[order] / count if count else 0.0 for order, count in order_counts.items()
    }
    total_order_count = sum(order_counts.values())
    overall_accuracy = (
        sum(order_correct.values()) / total_order_count if total_order_count else 0.0
    )
    sum_total = sum(sum_counts.values())
    pair_ratios = {
        key: value / pair_total if pair_total else 0.0 for key, value in pair_counts.items()
    }
    readable_order_accuracy = {
        "gold_first_accuracy": order_accuracy.get("res", 0.0),
        "negative_first_accuracy": order_accuracy.get("fan_1_res", 0.0),
    }
    sample_outcome_ratios = {
        "gold_wins_neither_order": (sum_counts[0] / sum_total if sum_total else 0.0),
        "gold_wins_exactly_one_order": (sum_counts[1] / sum_total if sum_total else 0.0),
        "gold_wins_both_orders": (sum_counts[2] / sum_total if sum_total else 0.0),
    }
    pair_consistency_ratios = {
        "both_orders_choose_gold": pair_ratios.get("both_correct", 0.0),
        "orders_disagree": pair_ratios.get("self_contradictory", 0.0),
        "both_orders_choose_negative": pair_ratios.get("both_wrong", 0.0),
    }
    return {
        "summary": {
            "overall_accuracy": overall_accuracy,
            "order_accuracy": order_accuracy,
            "readable_order_accuracy": readable_order_accuracy,
            "sum_ratios": {str(key): value / sum_total for key, value in sum_counts.items()} if sum_total else {},
            "sum_counts": dict(sum_counts),
            "sample_outcome_ratios": sample_outcome_ratios,
            "pair_ratios": pair_ratios,
            "pair_counts": dict(pair_counts),
            "pair_consistency_ratios": pair_consistency_ratios,
        },
        "per_sample": per_sample,
    }
