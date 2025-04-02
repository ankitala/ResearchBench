import os
import json
import csv

# Configuration
CLASSNAME_LIST = [
    "Cell Biology", "Chemistry", "Earth Science", "Material Science", "Physics",
    "Energy Science", "Environmental Science", "Biology", "Business", "Law", "Math", "Astronomy"
]
BASE_PATH = r"[REDACTED_PATH]\ranking"

def process_fan_1_res():
    """Process fan_1_res subfolders and generate fan_1_llm_ranking.csv for each category."""
    for name in CLASSNAME_LIST[:4]:  # Limited to first 4 as per original partial list
        opt_path = os.path.join(BASE_PATH, name)
        res_path = os.path.join(opt_path, "fan_1_res")
        results = []

        for model_folder in os.listdir(res_path):
            model_path = os.path.join(res_path, model_folder)
            if not os.path.isdir(model_path):
                continue

            rank_list = []
            count_files = 0

            for file_name in os.listdir(model_path):
                if file_name.startswith("ranking_res_") and file_name.endswith(".json"):
                    count_files += 1
                    file_path = os.path.join(model_path, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "Rank" in data and isinstance(data["Rank"], (int, float)):
                            rank_list.append(data["Rank"] / 16)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            avg_rank = sum(rank_list) / len(rank_list) if rank_list else None
            results.append({"model_name": model_folder, "avg_rank": avg_rank, "count": count_files})

        csv_file = os.path.join(opt_path, "fan_1_llm_ranking.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model_name", "avg_rank", "count"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_file}")

def aggregate_llm_ranking(output_file_name="avg_llm_ranking.csv", include_per_class=False):
    """Aggregate LLM ranking data across categories into a single CSV."""
    agg_data = {}

    for classname in CLASSNAME_LIST:
        csv_path = os.path.join(BASE_PATH, classname, "llm_ranking.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model_name"]
                try:
                    avg_rank = float(row["avg_rank"]) if row["avg_rank"] not in ("", None) else None
                    count = float(row["count"]) if row["count"] not in ("", None) else 0.0
                except Exception as e:
                    print(f"Error converting data in {csv_path}: {e}")
                    continue

                if model_name not in agg_data:
                    agg_data[model_name] = {"sum_weighted": 0.0, "total_count": 0.0}
                    if include_per_class:
                        for cname in CLASSNAME_LIST:
                            agg_data[model_name][cname] = None

                if avg_rank is not None:
                    agg_data[model_name]["sum_weighted"] += avg_rank * count
                agg_data[model_name]["total_count"] += count
                if include_per_class:
                    agg_data[model_name][classname] = avg_rank

    # Generate output data
    output_data = []
    for model_name, data in agg_data.items():
        total_count = data["total_count"]
        weighted_avg = data["sum_weighted"] / total_count if total_count != 0 else None
        row_data = {"model_name": model_name, "avg_rank": weighted_avg}
        if include_per_class:
            row_data.update({cname: data[cname] for cname in CLASSNAME_LIST})
        else:
            row_data["count3"] = total_count
        output_data.append(row_data)

    # Write to CSV
    output_csv_path = os.path.join(BASE_PATH, output_file_name)
    fieldnames = ["model_name", "avg_rank"] + (CLASSNAME_LIST if include_per_class else ["count3"])
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    print(f"Aggregated CSV saved to: {output_csv_path}")

def main():
    """Main function to process and aggregate ranking data."""
    # Process fan_1_res for limited categories
    process_fan_1_res()

    # Aggregate all categories into avg_llm_ranking.csv
    aggregate_llm_ranking("avg_llm_ranking.csv", include_per_class=False)

    # Aggregate with per-class details into avg_llm_ranking2.csv
    aggregate_llm_ranking("avg_llm_ranking2.csv", include_per_class=True)

if __name__ == "__main__":
    main()
