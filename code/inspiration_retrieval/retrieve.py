import os
import pandas as pd
import re
import subprocess

# Configuration
input_path = r"[REDACTED_PATH]\Biology\result.xlsx"
distance_path = r"[REDACTED_PATH]\spider\distance\Biology"
screening_script_path = r"[REDACTED_PATH]\MOOSE-Chem\inspiration_screening.py"
chem_annotation_path = r"[REDACTED_PATH]\Biology\result.xlsx"
overall_sheet_path = r"[REDACTED_PATH]\Biology\result_with_overall.xlsx"
python_executable = r"[REDACTED_PATH]\python.exe"
api_key = "[REDACTED_API_KEY]"

def normalize_doi(doi):
    """Normalize DOI by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', doi).lower()

def create_overall_sheet(input_path, output_path):
    """Copy input Excel content to a new 'Overall' sheet in output file."""
    df = pd.read_excel(input_path)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Overall", index=False)
    print(f"Overall sheet created at {output_path}")

def process_background_questions():
    # Create Overall sheet
    create_overall_sheet(input_path, overall_sheet_path)

    # Load Excel data
    df = pd.read_excel(input_path)
    max_no = df["No"].max()

    # Process each background question ID
    for background_question_id in range(max_no + 1):
        # Get DOI for current ID
        current_doi = df[df["No"] == background_question_id]["doi"].values[0]
        normalized_current_doi = normalize_doi(current_doi)

        # Find matching subfolder
        target_subfolder = next(
            (os.path.join(distance_path, subfolder) for subfolder in os.listdir(distance_path)
             if os.path.isdir(os.path.join(distance_path, subfolder)) and normalize_doi(subfolder) == normalized_current_doi),
            None
        )

        if not target_subfolder:
            print(f"No matching subfolder found for DOI: {current_doi}")
            continue

        # Define output and input paths
        output_dir = os.path.join(target_subfolder, "4omini_retrieve_res.json")
        title_abstract_path = os.path.join(target_subfolder, "merge.json")

        # Build command
        command = [
            python_executable, "-u", screening_script_path,
            "--model_name", "4omini",
            "--api_type", "0",
            "--api_key", api_key,
            "--chem_annotation_path", overall_sheet_path,
            "--output_dir", output_dir,
            "--title_abstract_all_insp_literature_path", title_abstract_path,
            "--if_use_background_survey", "0",
            "--if_use_strict_survey_question", "0",
            "--num_screening_window_size", "15",
            "--num_screening_keep_size", "3",
            "--num_round_of_screening", "4",
            "--if_save", "1",
            "--background_question_id", str(background_question_id),
            "--if_select_based_on_similarity", "0"
        ]

        # Execute command
        print(f"Executing command for background_question_id {background_question_id}...")
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Command executed successfully for background_question_id {background_question_id}.")
            print("Standard Output:", result.stdout)
            print("Standard Error:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for background_question_id {background_question_id}.")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)

if __name__ == "__main__":
    process_background_questions()
