import ast
from pathlib import Path

from researchbench.prompts import moosechem_instruction_prompts


def _load_vendor_instruction_prompts():
    source_path = Path(__file__).parents[1] / "src" / "researchbench" / "vendor" / "moosechem_utils.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    keep = []
    allowed = {
        "DISCIPLINE",
        "MUTATION_CUSTOM_GUIDE",
        "HYPTHESIS_GENERATION_CUSTOM_GUIDE",
        "instruction_prompts",
    }
    for node in module.body:
        names = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.FunctionDef):
            names = [node.name]
        if any(name in allowed for name in names):
            keep.append(node)
    slim = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(slim)
    namespace = {}
    exec(compile(slim, str(source_path), "exec"), namespace)
    return namespace["instruction_prompts"]


def test_moosechem_prompts_are_loaded_from_vendored_utils_verbatim() -> None:
    original = _load_vendor_instruction_prompts()
    modules = [
        "first_round_inspiration_screening",
        "coarse_hypothesis_generation_only_core_inspiration",
        "hypothesis_generation_with_feedback_only_core_inspiration",
        "hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration",
        "final_recombinational_mutation_hyp_gene_same_bkg_insp",
        "final_recombinational_mutation_hyp_gene_between_diff_inspiration",
        "additional_round_inspiration_screening",
        "four_aspects_checking",
        "four_aspects_self_numerical_evaluation",
        "eval_matched_score",
    ]
    for module_name in modules:
        more_info = 3 if module_name == "additional_round_inspiration_screening" else None
        assert list(moosechem_instruction_prompts(module_name, more_info=more_info)) == original(module_name, more_info=more_info)
