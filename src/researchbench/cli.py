"""ResearchBench command line interface."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .builders import build_all, validate_data_dir
from .clients import ModelClient, make_client
from .data_io import iter_jsonl, read_json, write_json, write_jsonl
from .generation import estimate_generation_model_calls, generation_score_prompt, parse_matched_score, run_generation_record
from .ranking import run_ranking_record, score_ranking
from .retrieve import run_retrieve_record, score_retrieve


RANKING_ORDER_ALIASES = {
    "res": "res",
    "gold-first": "res",
    "fan_1_res": "fan_1_res",
    "negative-first": "fan_1_res",
    "both": "both",
}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _progress(label: str, done: int, total: int, started_at: float) -> None:
    if total <= 0:
        return
    width = 28
    ratio = done / total
    filled = int(width * ratio)
    elapsed = max(time.time() - started_at, 1e-6)
    rate = done / elapsed if done else 0.0
    eta = (total - done) / rate if rate else 0.0
    bar = "#" * filled + "." * (width - filled)
    sys.stderr.write(f"\r{label} [{bar}] {done}/{total} {ratio:6.1%} | {rate:5.2f}/s | ETA {eta:5.1f}s")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


class ProgressTracker:
    def __init__(self, label: str, total: int, *, min_interval: float = 0.1) -> None:
        self.label = label
        self.total = max(total, 1)
        self.done = 0
        self.started_at = time.time()
        self.min_interval = min_interval
        self._last_rendered_at = 0.0
        self._last_rendered_done = -1
        self._last_rendered_total = -1
        self._lock = threading.Lock()
        self._render(force=True)

    def _render(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self.done < self.total and now - self._last_rendered_at < self.min_interval:
            return
        if not force and self.done == self._last_rendered_done:
            return
        _progress(self.label, self.done, self.total, self.started_at)
        self._last_rendered_at = now
        self._last_rendered_done = self.done
        self._last_rendered_total = self.total

    def advance(self, amount: int = 1) -> None:
        with self._lock:
            self.done += amount
            if self.done > self.total:
                self.total = self.done
            self._render(force=self.done >= self.total)

    def finish(self) -> None:
        with self._lock:
            if self.done < self.total:
                self.total = self.done or self.total
            if self.done != self._last_rendered_done or self.total != self._last_rendered_total:
                self._render(force=True)


class ProgressModelClient(ModelClient):
    def __init__(self, get_client: Callable[[], ModelClient], tracker: ProgressTracker, *, is_mock: bool = False) -> None:
        self._get_client = get_client
        self._tracker = tracker
        self.is_mock = is_mock

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        try:
            return self._get_client().generate(prompt, temperature=temperature)
        finally:
            self._tracker.advance()


def _run_with_progress(
    rows: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    label: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    total = len(rows)
    if total == 0:
        return []
    concurrency = max(1, concurrency)
    started_at = time.time()
    _progress(label, 0, total, started_at)
    results: list[dict[str, Any] | None] = [None] * total
    if concurrency == 1:
        for idx, row in enumerate(rows):
            try:
                results[idx] = worker(row)
            except Exception as exc:
                sample_id = row.get("sample_id", f"row-{idx}")
                raise RuntimeError(f"{label} failed on sample {sample_id}") from exc
            _progress(label, idx + 1, total, started_at)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_item = {executor.submit(worker, row): (idx, row) for idx, row in enumerate(rows)}
            done = 0
            for future in as_completed(future_to_item):
                idx, row = future_to_item[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    sample_id = row.get("sample_id", f"row-{idx}")
                    raise RuntimeError(f"{label} failed on sample {sample_id}") from exc
                done += 1
                _progress(label, done, total, started_at)
    return [item for item in results if item is not None]


def _run_ordered_concurrent(
    rows: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    label: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    total = len(rows)
    if total == 0:
        return []
    concurrency = max(1, concurrency)
    results: list[dict[str, Any] | None] = [None] * total
    if concurrency == 1:
        for idx, row in enumerate(rows):
            try:
                results[idx] = worker(row)
            except Exception as exc:
                sample_id = row.get("sample_id", f"row-{idx}")
                raise RuntimeError(f"{label} failed on sample {sample_id}") from exc
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_item = {executor.submit(worker, row): (idx, row) for idx, row in enumerate(rows)}
            for future in as_completed(future_to_item):
                idx, row = future_to_item[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    sample_id = row.get("sample_id", f"row-{idx}")
                    raise RuntimeError(f"{label} failed on sample {sample_id}") from exc
    return [item for item in results if item is not None]


def _thread_local_client_getter(model: str, api_key: str | None, base_url: str | None):
    local = threading.local()

    def get_client():
        client = getattr(local, "client", None)
        if client is None:
            client = make_client(model, api_key=api_key, base_url=base_url)
            local.client = client
        return client

    return get_client


def _format_ratio(value: float) -> str:
    return f"{value:.4f} ({value:.2%})"


def _print_retrieve_summary(summary: dict[str, Any]) -> None:
    label_names = {
        "gold": "Gold inspiration hit ratio",
        "negative_t1": "Negative tier 1 selection ratio",
        "negative_t2": "Negative tier 2 selection ratio",
        "negative_t3": "Negative tier 3 selection ratio",
        "negative_unknown": "Unknown-label candidate selection ratio",
    }
    round_names = {
        "round1": "Round 1: kept 15 of 75 candidates",
        "round2": "Round 2: kept 3 of 75 candidates",
    }
    print("Retrieve score summary")
    for round_id in ("round1", "round2"):
        if round_id not in summary:
            continue
        print(f"- {round_names.get(round_id, round_id)}")
        for label, value in sorted(summary[round_id].items()):
            name = label_names.get(label, label)
            print(f"  - {name}: {_format_ratio(float(value))}")


def _print_generation_summary(summary: dict[str, Any]) -> None:
    print("Hypothesis generation score summary")
    print(f"- Scored samples: {summary.get('count', 0)}")
    print(f"- Average matched score (0-5): {float(summary.get('average_score', 0.0)):.4f}")
    print(f"- Generation accuracy (average score / 5): {_format_ratio(float(summary.get('accuracy', 0.0)))}")


def _print_ranking_summary(summary: dict[str, Any]) -> None:
    order_names = {
        "res": "Gold-first order accuracy (gold is candidate 1)",
        "fan_1_res": "Negative-first order accuracy (gold is candidate 2)",
    }
    sum_names = {
        "0": "Gold wins in neither order",
        "1": "Gold wins in exactly one order",
        "2": "Gold wins in both orders",
    }
    pair_names = {
        "both_correct": "Both orders choose gold",
        "self_contradictory": "Orders disagree",
        "both_wrong": "Both orders choose negative",
    }
    print("Hypothesis ranking score summary")
    print(f"- Overall directional accuracy: {_format_ratio(float(summary.get('overall_accuracy', 0.0)))}")
    for order, value in summary.get("order_accuracy", {}).items():
        print(f"- {order_names.get(order, order)}: {_format_ratio(float(value))}")
    if summary.get("sum_ratios"):
        print("- Sample-level two-order outcomes")
        for key in ("0", "1", "2"):
            if key in summary["sum_ratios"]:
                count = summary.get("sum_counts", {}).get(int(key), summary.get("sum_counts", {}).get(key, 0))
                print(f"  - {sum_names[key]}: {_format_ratio(float(summary['sum_ratios'][key]))}; count={count}")
    if summary.get("pair_ratios"):
        print("- Pair-level order-consistency outcomes")
        for key in ("both_correct", "self_contradictory", "both_wrong"):
            if key in summary["pair_ratios"]:
                count = summary.get("pair_counts", {}).get(key, 0)
                print(f"  - {pair_names[key]}: {_format_ratio(float(summary['pair_ratios'][key]))}; count={count}")


def _print_build_summary(summary: dict[str, Any], out_dir: str | Path) -> None:
    print("Data build summary")
    print(f"- Papers: {summary.get('papers', 0)}")
    print(f"- Retrieve tasks: {summary.get('retrieve', 0)}")
    print(f"- Generation tasks: {summary.get('generation', 0)}")
    print(f"- Ranking tasks: {summary.get('ranking', 0)}")
    print(f"- Retrieve build issues: {summary.get('retrieve_issues', 0)}")
    print(f"- Ranking build issues: {summary.get('ranking_issues', 0)}")
    print(f"- Validation reports: {Path(out_dir) / 'validation'}")


def cmd_build_data(args: argparse.Namespace) -> int:
    summary = build_all(args.source_root, args.out)
    _print_build_summary(summary, args.out)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ok, issues = validate_data_dir(args.data_dir)
    if ok:
        print("Validation passed.")
        return 0
    report = Path(args.data_dir) / "validation" / "validation_issues.csv"
    print(f"Validation failed with {len(issues)} issue(s). See {report}.")
    return 1


def cmd_run_retrieve(args: argparse.Namespace) -> int:
    rows = _load_jsonl(args.data)
    get_client = _thread_local_client_getter(args.model, args.api_key, args.base_url)
    preds = _run_with_progress(
        rows,
        lambda row: {**run_retrieve_record(row, get_client()), "model": args.model},
        label="retrieve samples",
        concurrency=args.concurrency,
    )
    write_jsonl(args.out, preds)
    print(f"Wrote {len(preds)} retrieve predictions to {args.out}")
    return 0


def cmd_score_retrieve(args: argparse.Namespace) -> int:
    result = score_retrieve(_load_jsonl(args.pred), _load_jsonl(args.data))
    out = args.out or Path(args.pred).with_suffix(".score.json")
    write_json(out, result)
    _print_retrieve_summary(result["summary"])
    return 0


def cmd_run_generate(args: argparse.Namespace) -> int:
    rows = _load_jsonl(args.data)
    get_client = _thread_local_client_getter(args.model, args.api_key, args.base_url)
    estimated_calls = sum(
        estimate_generation_model_calls(
            row,
            num_mutations=args.num_mutations,
            num_itr_self_refine=args.num_itr_self_refine,
            max_inspiration_search_steps=args.max_inspiration_search_steps,
            num_screening_window_size=args.num_screening_window_size,
            num_screening_keep_size=args.num_screening_keep_size,
            recom_num_beam_size=args.recom_num_beam_size,
        )
        for row in rows
    )
    tracker = ProgressTracker("generation API calls", estimated_calls)

    def make_progress_client() -> ModelClient:
        return ProgressModelClient(get_client, tracker, is_mock=args.model == "mock")

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        return {
            **run_generation_record(
                row,
                make_progress_client(),
                num_mutations=args.num_mutations,
                num_itr_self_refine=args.num_itr_self_refine,
                max_inspiration_search_steps=args.max_inspiration_search_steps,
                use_background_survey=args.use_background_survey,
                num_screening_window_size=args.num_screening_window_size,
                num_screening_keep_size=args.num_screening_keep_size,
                recom_num_beam_size=args.recom_num_beam_size,
                inner_concurrency=args.inner_concurrency,
                client_factory=make_progress_client,
            ),
            "model": args.model,
        }

    try:
        preds = _run_ordered_concurrent(rows, worker, label="generate", concurrency=args.concurrency)
    finally:
        tracker.finish()
    write_jsonl(args.out, preds)
    print(f"Wrote {len(preds)} generation predictions to {args.out}")
    return 0


def cmd_score_generate(args: argparse.Namespace) -> int:
    predictions = _load_jsonl(args.pred)
    data = _load_jsonl(args.data)
    result = _score_generation_concurrent(
        predictions,
        data,
        judge_model=args.judge_model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
    )
    out = args.out or Path(args.pred).with_suffix(".score.json")
    write_json(out, result)
    _print_generation_summary(result["summary"])
    return 0


def cmd_run_rank(args: argparse.Namespace) -> int:
    rows = _load_jsonl(args.data)
    get_client = _thread_local_client_getter(args.model, args.api_key, args.base_url)
    order = RANKING_ORDER_ALIASES[args.order]
    preds = _run_with_progress(
        rows,
        lambda row: {**run_ranking_record(row, get_client(), order=order), "model": args.model},
        label="ranking samples",
        concurrency=args.concurrency,
    )
    write_jsonl(args.out, preds)
    print(f"Wrote {len(preds)} ranking predictions to {args.out}")
    return 0


def _score_generation_concurrent(
    predictions: list[dict[str, Any]],
    data: list[dict[str, Any]],
    *,
    judge_model: str,
    api_key: str | None,
    base_url: str | None,
    concurrency: int,
) -> dict[str, Any]:
    data_by_id = {row["sample_id"]: row for row in data}
    get_client = _thread_local_client_getter(judge_model, api_key, base_url)

    def worker(pred: dict[str, Any]) -> dict[str, Any]:
        record = data_by_id.get(pred.get("sample_id"))
        if not record:
            return {}
        score = pred.get("matched_score")
        reason = pred.get("matched_score_reason", "")
        if score is None:
            key_points = record.get("gold_key_points") or record["gold_hypothesis"]
            response = get_client().generate(
                generation_score_prompt(pred.get("final_hypothesis", ""), record["gold_hypothesis"], key_points),
                temperature=0.0,
            )
            score = parse_matched_score(response)
            reason = response
        if score is None:
            return {}
        score = float(score)
        return {
            "sample_id": pred["sample_id"],
            "matched_score": score,
            "accuracy": score / 5.0,
            "reason": reason,
        }

    per_sample = [
        item
        for item in _run_with_progress(predictions, worker, label="generation scoring", concurrency=concurrency)
        if item
    ]
    scores = [item["matched_score"] for item in per_sample]
    average_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "summary": {
            "count": len(scores),
            "average_score": average_score,
            "accuracy": average_score / 5.0,
        },
        "per_sample": per_sample,
    }


def cmd_score_rank(args: argparse.Namespace) -> int:
    result = score_ranking(_load_jsonl(args.pred))
    out = args.out or Path(args.pred).with_suffix(".score.json")
    write_json(out, result)
    _print_ranking_summary(result["summary"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchbench")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-data")
    build.add_argument("--source-root", required=True)
    build.add_argument("--out", required=True)
    build.set_defaults(func=cmd_build_data)

    validate = sub.add_parser("validate")
    validate.add_argument("data_dir")
    validate.set_defaults(func=cmd_validate)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", "--model-name", dest="model", required=True)
        p.add_argument("--api-key")
        p.add_argument("--base-url")
        p.add_argument("--concurrency", type=int, default=15, help="number of samples to evaluate concurrently")
        p.add_argument("--out", required=True)

    run_retrieve = sub.add_parser("run-retrieve")
    run_retrieve.add_argument("--data", required=True)
    add_model_args(run_retrieve)
    run_retrieve.set_defaults(func=cmd_run_retrieve)

    score_retrieve_parser = sub.add_parser("score-retrieve")
    score_retrieve_parser.add_argument("--pred", required=True)
    score_retrieve_parser.add_argument("--data", required=True)
    score_retrieve_parser.add_argument("--out")
    score_retrieve_parser.set_defaults(func=cmd_score_retrieve)

    run_generate = sub.add_parser("run-generate")
    run_generate.add_argument("--data", required=True)
    run_generate.add_argument("--num-mutations", type=int, default=2)
    run_generate.add_argument("--num-itr-self-refine", type=int, default=2)
    run_generate.add_argument("--max-inspiration-search-steps", type=int, default=3)
    run_generate.add_argument("--use-background-survey", action="store_true")
    run_generate.add_argument("--num-screening-window-size", type=int, default=12)
    run_generate.add_argument("--num-screening-keep-size", type=int, default=3)
    run_generate.add_argument("--recom-num-beam-size", type=int, default=15)
    run_generate.add_argument(
        "--inner-concurrency",
        type=int,
        default=1,
        help="within-sample concurrency for independent generation branches",
    )
    add_model_args(run_generate)
    run_generate.set_defaults(func=cmd_run_generate)

    score_generate_parser = sub.add_parser("score-generate")
    score_generate_parser.add_argument("--pred", required=True)
    score_generate_parser.add_argument("--data", required=True)
    score_generate_parser.add_argument("--judge-model", required=True)
    score_generate_parser.add_argument("--api-key")
    score_generate_parser.add_argument("--base-url")
    score_generate_parser.add_argument("--concurrency", type=int, default=15, help="number of samples to judge concurrently")
    score_generate_parser.add_argument("--out")
    score_generate_parser.set_defaults(func=cmd_score_generate)

    run_rank = sub.add_parser("run-rank")
    run_rank.add_argument("--data", required=True)
    run_rank.add_argument(
        "--order",
        choices=["gold-first", "negative-first", "res", "fan_1_res", "both"],
        default="both",
        help=(
            "comparison order: gold-first means gold is candidate 1; "
            "negative-first means the negative hypothesis is candidate 1; both runs both orders"
        ),
    )
    add_model_args(run_rank)
    run_rank.set_defaults(func=cmd_run_rank)

    score_rank_parser = sub.add_parser("score-rank")
    score_rank_parser.add_argument("--pred", required=True)
    score_rank_parser.add_argument("--out")
    score_rank_parser.set_defaults(func=cmd_score_rank)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
