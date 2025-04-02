import os
import json
import re
import csv
from collections import defaultdict
import random

# Configuration
res_path = r"[REDACTED_PATH]\Math\result.json"
distance_path = r"[REDACTED_PATH]\spider\distance\Math"
record_file = os.path.join(distance_path, "record.csv")
distance_2_3_csv_path = r"[REDACTED_PATH]\class\code\output20250107.csv"
class_name = "Math"

def normalize_doi(doi):
    """Normalize DOI by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', doi).lower()

def normalize_title(title):
    """Normalize title by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', title).lower()

def compare_dois_and_generate_record():
    # Load target DOIs from result.json
    with open(res_path, 'r', encoding='utf-8') as file:
        result_data = json.load(file)

    target_dois = {normalize_doi(item['doi']) for item in result_data if 'doi' in item}
    insufficient_dois = {normalize_doi(item['doi']) for item in result_data if 'doi' in item and item.get('sufficiency_tag') == 'no'}
    sufficient_dois = {normalize_doi(item['doi']) for item in result_data if 'doi' in item and item.get('sufficiency_tag') == 'yes'}

    # Load folder DOIs
    folder_dois = {normalize_doi(folder) for folder in os.listdir(distance_path) if os.path.isdir(os.path.join(distance_path, folder))}

    # Compare DOIs
    missing_dois = target_dois - folder_dois
    extra_dois = folder_dois - target_dois
    conflicting_dois = insufficient_dois & folder_dois

    # Write to record.csv
    with open(record_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["doi", "distance"])
        for doi in extra_dois | conflicting_dois | missing_dois:
            writer.writerow([doi, "no"])
        for doi in sufficient_dois & folder_dois:
            writer.writerow([doi, "yes"])

    # Print results
    for label, dois in [("Missing DOIs", missing_dois), ("Extra DOIs", extra_dois), ("Conflicting DOIs", conflicting_dois)]:
        print(f"{label}: {' '.join(dois) if dois else 'None'}")

def extract_non_class_data(csv_path, excluded_class):
    """Extract data from CSV excluding specified class, returning a dict by class type."""
    data = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get('class') != excluded_class:
                class_type = row.get('class', '').strip()
                cite = int(row.get('cite', 0)) if row.get('cite', '').isdigit() else 0
                data[class_type].append({
                    'title': row.get('title', '').strip(),
                    'abstract': row.get('abstract', '').strip(),
                    'cite': cite
                })
    return data

def save_json_to_folders(data, folder_path, filename):
    """Save JSON data to specified file in all subfolders."""
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, filename)
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

def load_inspiration_data(res_data, doi):
    """Load inspiration data for a given DOI from res_data."""
    normalized_doi = normalize_doi(doi)
    for item in res_data:
        if "doi" in item and normalize_doi(item["doi"]) == normalized_doi:
            return [[insp["title"], insp["abstract"]] for insp in item.get("inspiration", []) if "title" in insp and "abstract" in insp]
    return []

def clean_duplicates(data):
    """Remove duplicate titles from data list."""
    seen = set()
    return [item for item in data if len(item) == 2 and normalize_title(item[0]) not in seen and not seen.add(normalize_title(item[0]))]

def process_distance_file(file_path, existing_titles):
    """Process distance file, removing duplicates and excluding existing titles."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Unable to parse {file_path}, skipping.")
            return []

    if isinstance(data, dict):
        all_items = data.get("citations", []) + data.get("references", []) + data.get("semantics", [])
        formatted_data = [[item["title"], item["abstract"]] for item in all_items if "title" in item and "abstract" in item]
    else:
        formatted_data = data

    return [item for item in clean_duplicates(formatted_data) if normalize_title(item[0]) not in existing_titles]

def distribute_slots_evenly(total_slots, data1, data2, data3):
    """Distribute remaining slots evenly across three data lists."""
    selected1, selected2, selected3 = [], [], []
    while total_slots > 0:
        if data1 and total_slots > 0:
            selected1.append(data1.pop(0))
            total_slots -= 1
        if data2 and total_slots > 0:
            selected2.append(data2.pop(0))
            total_slots -= 1
        if data3 and total_slots > 0:
            selected3.append(data3.pop(0))
            total_slots -= 1
    return selected1, selected2, selected3

def merge_and_shuffle(subfolder):
    """Merge d0-d3.json files into merge.json with shuffled order."""
    files = [os.path.join(subfolder, f"d{i}.json") for i in range(4)]
    merged_data = []
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                merged_data.extend(json.load(file))

    random.shuffle(merged_data)
    merge_file = os.path.join(subfolder, "merge.json")
    with open(merge_file, 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, ensure_ascii=False, indent=4)
    print(f"Merged and shuffled data saved to {merge_file}.")

def main():
    # Step 1: Compare DOIs and generate record.csv
    compare_dois_and_generate_record()

    # Step 2: Extract and save non-class data
    class_data = extract_non_class_data(distance_2_3_csv_path, class_name)
    result = []
    for class_type, items in class_data.items():
        result.extend([[item['title'], item['abstract']] for item in sorted(items, key=lambda x: x['cite'], reverse=True)[:10]])
    save_json_to_folders(result, distance_path, "distance3.json")
    print(f"Non-{class_name} data processed and saved to subfolders in {distance_path}.")

    # Step 3: Process subfolders
    with open(res_path, 'r', encoding='utf-8') as file:
        res_data = json.load(file)

    subfolders = [os.path.join(distance_path, subfolder) for subfolder in os.listdir(distance_path) if os.path.isdir(os.path.join(distance_path, subfolder))]
    for subfolder in subfolders:
        doi = os.path.basename(subfolder)

        # Process d0.json
        inspiration_data = load_inspiration_data(res_data, doi)
        cleaned_inspiration = clean_duplicates(inspiration_data)
        d0_file = os.path.join(subfolder, "d0.json")
        with open(d0_file, 'w', encoding='utf-8') as file:
            json.dump(cleaned_inspiration, file, ensure_ascii=False, indent=4)

        existing_titles = set(normalize_title(item[0]) for item in cleaned_inspiration)
        d0_count = len(cleaned_inspiration)

        # Process d1, d2, d3
        d1_all_data = process_distance_file(os.path.join(subfolder, "distance1.json"), existing_titles)
        d2_all_data = process_distance_file(os.path.join(subfolder, "distance2.json"), existing_titles)
        d3_all_data = process_distance_file(os.path.join(subfolder, "distance3.json"), existing_titles)

        remaining_slots = max(75 - d0_count, 0)
        d1_data, d2_data, d3_data = distribute_slots_evenly(remaining_slots, d1_all_data, d2_all_data, d3_all_data)

        # Save d1, d2, d3
        for i, data in enumerate([d1_data, d2_data, d3_data], 1):
            with open(os.path.join(subfolder, f"d{i}.json"), 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        # Statistics
        counts = [d0_count, len(d1_data), len(d2_data), len(d3_data)]
        total_count = sum(counts)
        print(f"Subfolder: {subfolder}")
        print(f"d0: {counts[0]}, d1: {counts[1]}, d2: {counts[2]}, d3: {counts[3]} items. Total: {total_count}")
        if total_count != 75:
            print(f"WARNING: Total items ({total_count}) do not equal 75!")

        # Merge and shuffle
        merge_and_shuffle(subfolder)

if __name__ == "__main__":
    main()
