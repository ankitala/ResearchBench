import os
import pandas as pd
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Configuration
NAME_LIST = [
    "Cell Biology", "Chemistry", "Earth Science", "Material Science", "Physics",
    "Energy Science", "Environmental Science", "Biology", "Business", "Law", "Math", "Astronomy"
]
BASE_PATH = r"[REDACTED_PATH]\generation_merged"
API_TYPE = 0
API_KEY = "[REDACTED_API_KEY]"
MODEL_NAME = "4omini"
PYTHON_EXECUTABLE = r"[REDACTED_PATH]\python.exe"
SCREENING_SCRIPT_PATH = r"[REDACTED_PATH]\MOOSE-Chem\hypothesis_generation_bm.py"

def normalize_doi(doi):
    """Normalize DOI by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[\W_]', '', doi).lower()

def execute_commands(input_path, api_type, api_key, model_name, python_executable, screening_script_path):
    """Execute hypothesis generation commands for valid rows in the Excel file."""
    xlsx_path = os.path.join(input_path, "ranking_fake_result.xlsx")
    
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Error reading {xlsx_path}: {e}")
        return

    max_no = df["No"].max()

    for _, row in df.iterrows():
        no = row["No"]
        doi = row["doi"]
        if row.get("sufficiency tag") != "yes" or row.get("distance tag") != "yes" or row.get("inf tag") != "yes":
            continue

        normalized_doi = normalize_doi(doi)
        matching_folder = next(
            (os.path.join(input_path, folder) for folder in os.listdir(input_path)
             if os.path.isdir(os.path.join(input_path, folder)) and normalize_doi(folder) == normalized_doi),
            None
        )

        if not matching_folder:
            print(f"No matching folder found for DOI: {doi}")
            continue

        # Define file paths
        title_abstract_path = os.path.join(matching_folder, "merge.json")
        inspiration_dir = os.path.join(matching_folder, "ranking_fake_retrieve.json")
        output_dir = os.path.join(matching_folder, "ranking_generate_res_fake_retrieve.json")

        # Build command
        command = [
            python_executable, "-u", screening_script_path, "--model_name", model_name,
            "--api_type", str(api_type), "--api_key", api_key,
            "--chem_annotation_path", xlsx_path,
            "--if_use_strict_survey_question", "0", "--if_use_background_survey", "0",
            "--inspiration_dir", inspiration_dir,
            "--output_dir", output_dir,
            "--if_save", "1", "--if_load_from_saved", "0",
            "--if_use_gdth_insp", "1", "--idx_round_of_first_step_insp_screening", "0",
            "--num_mutations", "2", "--num_itr_self_refine", "2",
            "--num_self_explore_steps_each_line", "3", "--num_screening_window_size", "12",
            "--num_screening_keep_size", "3",
            "--if_mutate_inside_same_bkg_insp", "1", "--if_mutate_between_diff_insp", "1",
            "--if_self_explore", "0",
            "--if_consider_external_knowledge_feedback_during_second_refinement", "0",
            "--inspiration_ids", "-1",
            "--recom_num_beam_size", "15",
            "--self_explore_num_beam_size", "15",
            "--max_inspiration_search_steps", "3",
            "--background_question_id", str(no),
            "--title_abstract_all_insp_literature_path", title_abstract_path
        ]

        # Execute command with real-time output
        print(f"Executing command for No: {no}, DOI: {doi}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        for line in iter(process.stderr.readline, ''):
            print(line, end='')
        process.stdout.close()
        process.stderr.close()
        process.wait()

def run_in_parallel():
    """Run command execution in parallel across all categories."""
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                execute_commands, os.path.join(BASE_PATH, class_name), API_TYPE, API_KEY,
                MODEL_NAME, PYTHON_EXECUTABLE, SCREENING_SCRIPT_PATH
            )
            for class_name in NAME_LIST
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel execution: {e}")

if __name__ == "__main__":
    run_in_parallel()
