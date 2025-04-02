import os
import json
import pandas as pd
import re

# Configuration
BASE_PATH = r"[REDACTED_PATH]\generation_merged"
NAME_LIST = [
    "Cell Biology", "Chemistry", "Earth Science", "Material Science", "Physics",
    "Energy Science", "Environmental Science", "Biology", "Business", "Law", "Math", "Astronomy"
]

def load_json(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def normalize_doi(doi):
    """Normalize DOI by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[^\w]', '', doi).lower()

def process_folder(folder_path):
    """Process folder to generate ranking_fake_retrieve.json from 4o_retrieve_res.json and ture_retrieve.json."""
    file_4o = os.path.join(folder_path, "4o_retrieve_res.json")
    file_true = os.path.join(folder_path, "ture_retrieve.json")
    output_file = os.path.join(folder_path, "ranking_fake_retrieve.json")

    if not (os.path.isfile(file_4o) and os.path.isfile(file_true)):
        return

    # Process 4o_retrieve_res.json
    data_4o = load_json(file_4o)
    key_4o = list(data_4o[0].keys())[0]
    wait_list_title_abstract = [pair for small_list in reversed(data_4o[0][key_4o]) for pair in reversed(small_list)]

    # Process ture_retrieve.json
    data_true = load_json(file_true)
    key_true = list(data_true[0].keys())[0]
    true_titles = {pair[0] for pair in data_true[0][key_true][0] if pair and isinstance(pair, list) and len(pair) >= 1}

    # Select 3 unique pairs excluding true titles
    selected_pairs = [pair for pair in wait_list_title_abstract if pair[0] not in true_titles][:3]
    if len(selected_pairs) < 3:
        print(f"Folder {folder_path} has fewer than 3 valid pairs, skipping.")
        return

    # Construct new data structure
    new_data = [
        {key_true: [selected_pairs]},
        {"backgorund_question": [[1.0, 1.0]]}
    ]
    save_json(new_data, output_file)
    print(f"Generated {output_file} in {folder_path}")

def process_ranking_fake_retrieve(json_path):
    """Extract three title-abstract pairs from ranking_fake_retrieve.json."""
    data = load_json(json_path)
    if not isinstance(data, list) or len(data) < 1 or not isinstance(data[0], dict):
        print(f"Unexpected format in {json_path}")
        return []

    key, value = next(iter(data[0].items()))
    sublists = value[0] if isinstance(value, list) and value else []
    if len(sublists) < 3:
        print(f"Not enough pairs in {json_path}")
        return []

    return [(sublist[0], sublist[1]) for sublist in sublists[:3] if isinstance(sublist, list) and len(sublist) == 2]

def update_excel_with_ranking_fake():
    """Update Excel files with data from ranking_fake_retrieve.json."""
    for name in NAME_LIST:
        opt_path = os.path.join(BASE_PATH, name)
        excel_input_path = os.path.join(opt_path, "result_with_overall.xlsx")
        excel_output_path = os.path.join(opt_path, "ranking_fake_result.xlsx")

        print(f"\nProcessing category: {name}")
        if not os.path.isfile(excel_input_path):
            print(f"Excel file {excel_input_path} does not exist, skipping.")
            continue

        # Load Excel
        try:
            df = pd.read_excel(excel_input_path, sheet_name='Overall', dtype=str)
        except Exception as e:
            print(f"Error reading {excel_input_path}: {e}, skipping.")
            continue

        if 'doi' not in df.columns:
            print(f"No 'doi' column in {excel_input_path}, skipping.")
            continue

        df['normalized_doi'] = df['doi'].apply(normalize_doi)
        doi_to_index = {doi: idx for idx, doi in enumerate(df['normalized_doi'])}
        updated_count = 0

        # Process subdirectories
        for subdir in os.listdir(opt_path):
            subdir_path = os.path.join(opt_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            json_path = os.path.join(subdir_path, "ranking_fake_retrieve.json")
            if not os.path.isfile(json_path):
                continue

            normalized_subdir_doi = normalize_doi(subdir)
            if normalized_subdir_doi not in doi_to_index:
                print(f"No matching DOI for {subdir} in Excel.")
                continue

            row_idx = doi_to_index[normalized_subdir_doi]
            pairs = process_ranking_fake_retrieve(json_path)
            if len(pairs) != 3:
                print(f"Insufficient pairs in {json_path}.")
                continue

            # Update DataFrame
            for i, (title, abstract) in enumerate(pairs, 1):
                df.at[row_idx, f'Inspiration paper {i} title'] = title
                df.at[row_idx, f'Relation between the main inspiration and the inspiration paper {i}'] = abstract
            updated_count += 1
            print(f"Updated DOI: {subdir}")

        print(f"Total updates for '{name}': {updated_count}")

        # Ensure required columns exist
        required_columns = [
            f'Inspiration paper {i} title' for i in range(1, 4)
        ] + [
            f'Relation between the main inspiration and the inspiration paper {i}' for i in range(1, 4)
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        # Save updated Excel
        df.drop(columns=['normalized_doi'], inplace=True)
        try:
            with pd.ExcelWriter(excel_output_path) as writer:
                df.to_excel(writer, sheet_name='Overall', index=False)
            print(f"Saved {excel_output_path}")
        except Exception as e:
            print(f"Error saving {excel_output_path}: {e}")

def main():
    """Main function to process folders and update Excel files."""
    for name in NAME_LIST:
        opt_path = os.path.join(BASE_PATH, name)
        if not os.path.exists(opt_path):
            print(f"Path {opt_path} does not exist, skipping.")
            continue

        # Process each subfolder to generate ranking_fake_retrieve.json
        for root, dirs, _ in os.walk(opt_path):
            for d in dirs:
                folder_path = os.path.join(root, d)
                try:
                    process_folder(folder_path)
                except Exception as e:
                    print(f"Error processing {folder_path}: {e}")
            break  # Only process top-level subfolders

    # Update Excel files with ranking_fake_retrieve.json data
    update_excel_with_ranking_fake()

if __name__ == "__main__":
    main()
