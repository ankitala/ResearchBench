import os
import json
import csv

name_list = ["Cell Biology"]
father_path = r"[REDACTED_PATH]\generation_merged"

for name in name_list:
    opt_path = os.path.join(father_path, name)

    # Iterate through subfolders in opt_path
    for subfolder in os.listdir(opt_path):
        subfolder_path = os.path.join(opt_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Find JSON files starting with "eval_res_" and ending with ".json"
        eval_files = [f for f in os.listdir(subfolder_path)
                      if f.startswith("eval_res_") and f.endswith(".json")]

        if len(eval_files) < 10:
            print(f"Skipping folder {subfolder_path}, only {len(eval_files)} matching files.")
            continue

        print(f"Processing folder: {subfolder_path}")

        for filename in eval_files:
            file_path = os.path.join(subfolder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cur_data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # Extract element at index 2
            try:
                cur_matched_insp_hyp_collection = cur_data[2]
            except IndexError:
                print(f"File {file_path} missing data at index 2.")
                continue

            # Group hypotheses by round
            new_data = {}
            for record in cur_matched_insp_hyp_collection:
                try:
                    cur_hyp = record[0]
                    cur_matched_score = record[7]
                    cur_round_id = record[9]
                except IndexError:
                    print(f"Invalid record format in {file_path}: {record}")
                    continue

                round_key = f"round {cur_round_id}"
                if round_key not in new_data:
                    new_data[round_key] = []
                new_data[round_key].append({
                    "content": cur_hyp,
                    "ground_truth_score": cur_matched_score
                })

            # Sort hypotheses within each round by score
            for round_key in new_data:
                sorted_hyps = sorted(new_data[round_key], key=lambda x: x["ground_truth_score"], reverse=True)
                new_hyp_dict = {f"hypothesis {idx}": hyp for idx, hyp in enumerate(sorted_hyps, start=1)}
                new_data[round_key] = new_hyp_dict

            # Sort rounds by number in descending order
            sorted_round_keys = sorted(new_data.keys(), key=lambda k: int(k.split()[1]), reverse=True)
            new_data_sorted = {key: new_data[key] for key in sorted_round_keys}

            # Save to new JSON file with "score_res_" prefix
            new_filename = "score_res_" + filename[len("eval_res_"):]
            new_file_path = os.path.join(subfolder_path, new_filename)
            try:
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data_sorted, f, ensure_ascii=False, indent=4)
                print(f"File saved: {new_file_path}")
            except Exception as e:
                print(f"Error writing file {new_file_path}: {e}")

# Model suffix to name mapping
model_mapping = {
    "4o.json": "gpt-4o-2024-11-20",
    "4omini.json": "gpt-4o-mini-2024-07-18",
    "claude35haiku.json": "claude-3-5-haiku-20241022",
    "claude35sonnet.json": "claude-3-5-sonnet-20241022",
    "deepseek.json": "DeepSeek-V3",
    "gemini2flash.json": "gemini-2.0-flash-exp",
    "gemini2flashthinking.json": "gemini-2.0-flash-thinking-exp",
    "llama318b.json": "Meta-Llama-3.1-8B-Instruct",
    "llama321b.json": "Meta-Llama-3.2-1B-Instruct",
    "llama3170b.json": "Meta-Llama-3.1-70B-Instruct",
    "qwenplus.json": "qwen-plus-2024-11-25",
    "qwenturbo.json": "qwen-turbo-2024-11-01"
}

for name in name_list:
    opt_path = os.path.join(father_path, name)
    print(f"Processing category: {name}, path: {opt_path}")

    if not os.path.isdir(opt_path):
        print(f"Directory not found: {opt_path}, skipping.")
        continue

    # Track model scores: {model_name: [total_score, count]}
    model_scores = {}

    for subfolder in os.listdir(opt_path):
        subfolder_path = os.path.join(opt_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            if not (filename.startswith("score_res_") and filename.endswith(".json")):
                continue

            suffix = filename[len("score_res_"):]
            if suffix not in model_mapping:
                continue
            model_name = model_mapping[suffix]

            file_path = os.path.join(subfolder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Find the final round
            round_numbers = [int(key.split()[1]) for key in data.keys() if key.startswith("round ")]
            if not round_numbers:
                print(f"No round info in {file_path}")
                continue
            final_round_key = f"round {max(round_numbers)}"

            if final_round_key not in data:
                print(f"Key {final_round_key} not found in {file_path}")
                continue

            final_round_data = data[final_round_key]
            if not isinstance(final_round_data, dict) or not final_round_data:
                print(f"No hypothesis data in {final_round_key} of {file_path}")
                continue

            # Calculate average score for valid hypotheses
            total_score = 0.0
            hyp_count = 0
            for hyp_key, hyp_data in final_round_data.items():
                if "content" not in hyp_data or len(hyp_data["content"]) < 20:
                    continue
                if "ground_truth_score" not in hyp_data or float(hyp_data["ground_truth_score"]) == 0:
                    continue
                total_score += float(hyp_data["ground_truth_score"])
                hyp_count += 1

            if hyp_count == 0:
                print(f"No valid hypotheses in {file_path}")
                continue

            avg_score = total_score / hyp_count
            if model_name not in model_scores:
                model_scores[model_name] = [0.0, 0]
            model_scores[model_name][0] += avg_score
            model_scores[model_name][1] += 1

    # Write results to CSV
    output_csv = os.path.join(opt_path, "model_generate_score.csv")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["model", "avg_score", "count", "totle_score"])
            for model, (total, count) in model_scores.items():
                overall_avg = total / count if count > 0 else 0.0
                writer.writerow([model, overall_avg, count, total])
        print(f"CSV saved to: {output_csv}")
    except Exception as e:
        print(f"Error writing CSV {output_csv}: {e}")


name_list = [
    "Astronomy", "Biology", "Business", "Cell Biology", "Chemistry",
    "Earth Science", "Energy Science", "Environmental Science", "Law",
    "Material Science", "Math", "Physics"
]
father_path = r"[REDACTED_PATH]\generation_score"

# Aggregate model data: {model: {"count": total_count, "totle_score": total_score}}
aggregate = {}

for name in name_list:
    category_path = os.path.join(father_path, name)
    csv_path = os.path.join(category_path, "model_generate_score.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found, skipping {name}")
        continue

    print(f"Reading file: {csv_path}")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model = row["model"].strip()
                try:
                    count = int(row["count"])
                    totle_score = float(row["totle_score"])
                except Exception as e:
                    print(f"Error converting data in {row}: {e}")
                    continue
                if model not in aggregate:
                    aggregate[model] = {"count": 0, "totle_score": 0.0}
                aggregate[model]["count"] += count
                aggregate[model]["totle_score"] += totle_score
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

# Write aggregated results to final CSV
final_csv = os.path.join(father_path, "model_generate_score.csv")
try:
    with open(final_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "avg_score", "count", "totle_score"])
        for model, data in aggregate.items():
            total_count = data["count"]
            total_score = data["totle_score"]
            overall_avg = total_score / total_count if total_count > 0 else 0.0
            writer.writerow([model, overall_avg, total_count, total_score])
    print(f"Final CSV saved to: {final_csv}")
except Exception as e:
    print(f"Error writing final CSV {final_csv}: {e}")
