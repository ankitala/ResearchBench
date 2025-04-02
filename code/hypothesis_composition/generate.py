import os
import pandas as pd
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

def normalize_doi(doi):
    return re.sub(r'[\W_]', '', doi).lower()

def execute_commands(input_path, api_type, api_key, model_name, python_executable, screening_script_path):
    xlsx_path = os.path.join(input_path, "result_with_overall.xlsx")
    df = pd.read_excel(xlsx_path)
    max_no = df["No"].max()

    for _, row in df.iterrows():
        no = row["No"]
        doi = row["doi"]
        sufficiency_tag = row["sufficiency tag"]
        distance_tag = row["distance tag"]
        inf_tag = row["inf tag"]

        # Skip if tags are not all "yes"
        if sufficiency_tag != "yes" or distance_tag != "yes" or inf_tag != "yes":
            continue

        normalized_doi = normalize_doi(doi)
        matching_folder = None

        # Find matching folder by normalized DOI
        for folder in os.listdir(input_path):
            folder_path = os.path.join(input_path, folder)
            if os.path.isdir(folder_path) and normalize_doi(folder) == normalized_doi:
                matching_folder = folder_path
                break

        if not matching_folder:
            print(f"No matching folder found for DOI: {doi}")
            continue

        title_abstract_path = os.path.join(matching_folder, "merge.json")
        inspiration_dir = os.path.join(matching_folder, "true_retrieve.json")
        output_dir = os.path.join(matching_folder, "generate_res_bf.json")

        # Build command for subprocess
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
            f"--background_question_id", str(no),
            f"--title_abstract_all_insp_literature_path", title_abstract_path
        ]

        print(f"Executing command for No: {no}, DOI: {doi}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        for line in iter(process.stderr.readline, ''):
            print(line, end='')

        process.stdout.close()
        process.stderr.close()
        process.wait()

def run_in_parallel(name_list, api_type, api_key, model_name, python_executable, screening_script_path):
    with ThreadPoolExecutor() as executor:
        futures = []
        for class_name in name_list:
            input_path = r"[REDACTED_PATH]\\generation\\" + class_name
            futures.append(
                executor.submit(
                    execute_commands, input_path, api_type, api_key, model_name, python_executable, screening_script_path
                )
            )
        # Wait for all tasks to complete
        for future in futures:
            future.result()

name_list = ["test_class1", "test_class2"]
api_type = 0
api_key = "[REDACTED_API_KEY]"
model_name = "4omini"
python_executable = r"[REDACTED_PATH]\\python.exe"
screening_script_path = r"[REDACTED_PATH]\\hypothesis_generation_bm.py"

run_in_parallel(name_list, api_type, api_key, model_name, python_executable, screening_script_path)
