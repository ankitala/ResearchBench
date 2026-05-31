"""Prompt templates for ResearchBench.

The retrieve and generation prompts are loaded from the vendored
MOOSE-Chem ``Method/utils.py`` snapshot to preserve exact prompt text.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any


def _load_moosechem_prompt_namespace() -> dict[str, Any]:
    source_path = Path(__file__).parent / "vendor" / "moosechem_utils.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    keep: list[ast.stmt] = []
    allowed = {
        "DISCIPLINE",
        "MUTATION_CUSTOM_GUIDE",
        "HYPTHESIS_GENERATION_CUSTOM_GUIDE",
        "instruction_prompts",
    }
    for node in module.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.FunctionDef):
            names = [node.name]
        if any(name in allowed for name in names):
            keep.append(node)
    slim_module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(slim_module)
    namespace: dict[str, Any] = {}
    exec(compile(slim_module, str(source_path), "exec"), namespace)
    return namespace


@lru_cache(maxsize=64)
def moosechem_instruction_prompts(module_name: str, more_info: int | None = None) -> tuple[str, ...]:
    """Return the original MOOSE-Chem prompt pieces for ``module_name``."""

    namespace = _load_moosechem_prompt_namespace()
    prompts = namespace["instruction_prompts"](module_name, more_info=more_info)
    return tuple(prompts)


def format_candidates(candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, candidate in enumerate(candidates):
        blocks.append(
            "Next we will introduce inspiration candidate {}. Title: {}; Abstract: {}. "
            "The introduction of inspiration candidate {} has come to an end.\n".format(
                idx,
                candidate.get("title", ""),
                candidate.get("abstract", ""),
                idx,
            )
        )
    return "".join(blocks)


def retrieve_prompt(
    research_question: str,
    background_survey: str,
    candidates: list[dict[str, Any]],
    *,
    use_background_survey: bool = False,
    select_based_on_similarity: bool = False,
) -> str:
    module_name = (
        "first_round_inspiration_screening_only_based_on_semantic_similarity"
        if select_based_on_similarity
        else "first_round_inspiration_screening"
    )
    prompts = moosechem_instruction_prompts(module_name)
    survey = background_survey if use_background_survey else "Survey not provided. Please overlook the survey."
    return prompts[0] + research_question + prompts[1] + survey + prompts[2] + format_candidates(candidates) + prompts[3]


def background_survey_text(background_survey: str, *, use_background_survey: bool) -> str:
    if use_background_survey:
        return background_survey
    return "Survey not provided. Please overlook the survey."


def core_inspiration_text(inspiration: dict[str, Any]) -> str:
    return "title: {}; abstract: {}; one of the potential reasons on why this inspiration could be helpful: {}.".format(
        inspiration.get("title", ""),
        inspiration.get("abstract", ""),
        inspiration.get("reason", inspiration.get("insp", "Not provided yet.")),
    )


def generation_initial_prompt(
    research_question: str,
    background_survey: str,
    inspiration: dict[str, Any],
) -> str:
    prompts = moosechem_instruction_prompts("coarse_hypothesis_generation_only_core_inspiration")
    return prompts[0] + research_question + prompts[1] + background_survey + prompts[2] + core_inspiration_text(inspiration) + prompts[3]


def generation_feedback_prompt(hypothesis: str, reasoning: str) -> str:
    prompts = moosechem_instruction_prompts("four_aspects_checking")
    hypothesis_prompt = "hypothesis: {}; reasoning process: {}.".format(hypothesis, reasoning)
    return prompts[0] + hypothesis_prompt + prompts[1]


def generation_refine_prompt(
    research_question: str,
    background_survey: str,
    inspiration: dict[str, Any],
    previous_hypothesis: str,
    feedback: str,
) -> str:
    prompts = moosechem_instruction_prompts("hypothesis_generation_with_feedback_only_core_inspiration")
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(inspiration)
        + prompts[3]
        + previous_hypothesis
        + prompts[4]
        + feedback
        + prompts[5]
    )


def generation_distinct_mutation_prompt(
    research_question: str,
    background_survey: str,
    inspiration: dict[str, Any],
    previous_hypotheses: list[str],
) -> str:
    prompts = moosechem_instruction_prompts("hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration")
    previous = "".join(f"Next is previous hypothesis {idx}: {hypothesis}.\n" for idx, hypothesis in enumerate(previous_hypotheses))
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(inspiration)
        + prompts[3]
        + previous
        + prompts[4]
    )


def generation_same_inspiration_recombine_prompt(
    research_question: str,
    background_survey: str,
    inspiration: dict[str, Any],
    hypotheses: list[str],
) -> str:
    prompts = moosechem_instruction_prompts("final_recombinational_mutation_hyp_gene_same_bkg_insp")
    previous = "".join(f"Next is previous hypothesis {idx}: {hypothesis}.\n" for idx, hypothesis in enumerate(hypotheses))
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(inspiration)
        + prompts[3]
        + previous
        + prompts[4]
    )


def generation_same_inspiration_recombine_refine_prompt(
    research_question: str,
    background_survey: str,
    inspiration: dict[str, Any],
    hypotheses: list[str],
    previous_hypothesis: str,
    feedback: str,
) -> str:
    prompts = moosechem_instruction_prompts("final_recombinational_mutation_hyp_gene_same_bkg_insp_with_feedback")
    previous = "".join(f"Next is previous hypothesis {idx}: {hypothesis}.\n" for idx, hypothesis in enumerate(hypotheses))
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(inspiration)
        + prompts[3]
        + previous
        + prompts[4]
        + previous_hypothesis
        + prompts[5]
        + feedback
        + prompts[6]
    )


def generation_between_inspiration_recombine_prompt(
    research_question: str,
    background_survey: str,
    core_inspiration: dict[str, Any],
    core_hypothesis: str,
    other_inspiration: dict[str, Any],
    other_hypothesis: str,
) -> str:
    prompts = moosechem_instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration")
    other = (
        "The selected complementary inspiration has title: {}, and abstract: {}. "
        "This complementary inspiration can lead to the hypothesis: {}. This hypothesis could be useful "
        "to understand how this complementary inspiration can be helpful."
    ).format(other_inspiration.get("title", ""), other_inspiration.get("abstract", ""), other_hypothesis)
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(core_inspiration)
        + prompts[3]
        + core_hypothesis
        + prompts[4]
        + other
        + prompts[5]
    )


def generation_between_inspiration_recombine_refine_prompt(
    research_question: str,
    background_survey: str,
    core_inspiration: dict[str, Any],
    core_hypothesis: str,
    other_inspiration: dict[str, Any],
    other_hypothesis: str,
    previous_hypothesis: str,
    feedback: str,
) -> str:
    prompts = moosechem_instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration_with_feedback")
    other = (
        "The selected complementary inspiration has title: {}, and abstract: {}. "
        "This complementary inspiration can lead to the hypothesis: {}. This hypothesis could be useful "
        "to understand how this complementary inspiration can be helpful."
    ).format(other_inspiration.get("title", ""), other_inspiration.get("abstract", ""), other_hypothesis)
    return (
        prompts[0]
        + research_question
        + prompts[1]
        + background_survey
        + prompts[2]
        + core_inspiration_text(core_inspiration)
        + prompts[3]
        + core_hypothesis
        + prompts[4]
        + other
        + prompts[5]
        + previous_hypothesis
        + prompts[6]
        + feedback
        + prompts[7]
    )


def generation_self_eval_prompt(hypothesis: str) -> str:
    prompts = moosechem_instruction_prompts("four_aspects_self_numerical_evaluation")
    return prompts[0] + "hypothesis: {}.".format(hypothesis) + prompts[1]


def generation_score_prompt(proposed: str, gold: str, key_points: str) -> str:
    prompts = moosechem_instruction_prompts("eval_matched_score")
    return prompts[0] + proposed + prompts[1] + gold + prompts[2] + key_points + prompts[3]


def generation_additional_inspiration_screening_prompt(
    research_question: str,
    background_survey: str,
    core_inspiration: dict[str, Any],
    core_hypothesis: str,
    other_mutations: list[dict[str, Any]],
    *,
    keep_size: int,
) -> str:
    prompts = moosechem_instruction_prompts("additional_round_inspiration_screening", more_info=keep_size)
    core = "Title: {}; Abstract: {}.".format(
        core_inspiration.get("title", ""),
        core_inspiration.get("abstract", ""),
    )
    candidates = ""
    for idx, mutation in enumerate(other_mutations):
        candidates += (
            "Next we will introduce potential inspiration candidate {}. Title: {}; Abstract: {}. "
            "This inspiration has been leveraged to generate hypothesis for the given background question. "
            "The hypothesis is: {}. \n"
        ).format(
            idx,
            mutation.get("title", ""),
            mutation.get("abstract", ""),
            mutation.get("hypothesis", ""),
        )
    return prompts[0] + research_question + prompts[1] + background_survey + prompts[2] + core + prompts[3] + core_hypothesis + prompts[4] + candidates + prompts[5]


def ranking_prompt(research_question: str, candidate_1: str, candidate_2: str) -> str:
    return (
        "You are assisting scientists with their research. Given a research question and two research "
        "hypothesis candidates proposed by large language models, predict which hypothesis is better. By "
        "'better', we mean more valid and effective for the research question.\n\n"
        "(1) Neither hypothesis has been tested experimentally. Do not believe descriptions of expected "
        "performance; focus on technical content.\n\n"
        "(2) Focus on whether the general direction or major components are more effective. Additional "
        "detail, complexity, or breadth is neither an advantage nor a disadvantage by itself.\n\n"
        f"The research question is: {research_question}\n"
        f"Research hypothesis candidate 1 is: {candidate_1}\n"
        f"Research hypothesis candidate 2 is: {candidate_2}\n\n"
        "Now predict which hypothesis will be more effective if tested in real experiments. Use the response "
        "format:\n**Selection of research hypothesis candidate**: candidate 1 or candidate 2\n"
    )
