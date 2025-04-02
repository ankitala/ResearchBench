from openai import OpenAI
import time
import re
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# Configuration
API_KEY = "[REDACTED_API_KEY]"
BASE_URL = "https://api.claudeshop.top/v1"
MODEL_NAME = "gpt-4o"
CLASSNAME = "test"
OPT_PATH = r"[REDACTED_PATH]\ranking\\" + CLASSNAME
SAVED_PATH = r"[REDACTED_PATH]\ranking\\" + CLASSNAME + "\\res"
CONCURRENCY_NUM = 15

PROMPT_FOR_COMPARE = """You are assisting scientists with their research. Given a research question and two research hypothesis candidates proposed by large language models, your task is to predict which hypothesis is a better research hypothesis. By 'better', we mean the hypothesis is more valid and effective for the research question. 
Please note:
(1) Neither hypothesis has been tested experimentally. Ignore any described expected performance and focus only on technical content to predict effectiveness in real experiments.
(2) Focus on the core idea's effectiveness, not additional details or complexity.
The research question is: {}
Research hypothesis candidate 1 is: {}
Research hypothesis candidate 2 is: {}
Now, predict which hypothesis will be more effective if tested in real experiments. Use this format:
**Analysis**:
**Selection of research hypothesis candidate**: candidate 1 or candidate 2
"""

PATTERN = r"\*\*Selection\s+of\s+research\s+hypothesis\s+candidate\*\*\s*[:：]\s*candidate\s*(\d+)"

def get_llm_response(context, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME):
    """Get response from LLM API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    message_text = [{"role": "user", "content": context}]
    max_retries = 3000000

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name, messages=message_text, temperature=0.7, top_p=0.95,
                frequency_penalty=0, presence_penalty=0, stop=None
            )
            result = completion.choices[0].message.content.strip()
            print("result:\n", result, "\n\n")
            return result
        except Exception as e:
            error_message = str(e)
            if any(keyword in error_message for keyword in ["maximum context length", "maximum length", "context length", "string too long", "length"]):
                return "string too long"
            print(f"API error, Attempt {attempt + 1} failed: {error_message}")
            time.sleep(1)
    raise Exception("Failed to get a valid result after max retries.")

def compare_candidate(candidate_hypothesis, background_question, main_hypothesis, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME, max_retries=10):
    """Compare candidate hypothesis with main hypothesis using LLM."""
    prompt = PROMPT_FOR_COMPARE.format(background_question, main_hypothesis, candidate_hypothesis)
    for attempt in range(max_retries):
        response = get_llm_response(prompt, api_key, base_url, model_name)
        match = re.search(PATTERN, response)
        if match and (selection := int(match.group(1))) in [1, 2]:
            print(f"[compare_candidate] Success (attempt {attempt + 1}): Selected candidate {selection}")
            return selection
        print(f"[compare_candidate] Invalid format (attempt {attempt + 1}), retrying...")
    print("[compare_candidate] Max retries reached, defaulting to candidate 1")
    return 1

def process_file(file_path, concurrency_num=CONCURRENCY_NUM, saved_path=SAVED_PATH, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME):
    """Process a single JSON file and compute ranking."""
    print(f"\n========== Processing file: {file_path} ==========")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[process_file] Failed to read {file_path}: {e}")
        return

    # Extract data
    background_question = data.get("Background Question", "")
    main_hypothesis = data.get("Main hypothesis", "")
    combined_list = data.get("fake generate hypothesis", []) + data.get("model generate hypothesis", [])
    total_candidates = len(combined_list)
    print(f"[process_file] Total candidates to compare: {total_candidates}")

    # Compute rank
    rank_count = 16
    with ThreadPoolExecutor(max_workers=concurrency_num) as executor:
        futures = {executor.submit(compare_candidate, candidate, background_question, main_hypothesis, api_key, base_url, model_name): candidate for candidate in combined_list}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                if future.result() == 2:
                    rank_count -= 1
                print(f"[process_file] Processed {i}/{total_candidates} candidates")
            except Exception as e:
                print(f"[process_file] Error processing candidate: {e}")

    print(f"[process_file] Final Rank for {file_path}: {rank_count}")

    # Save results
    output_data = {
        "Background Question": background_question,
        "Main hypothesis": main_hypothesis,
        "Rank": rank_count,
        "fake generate hypothesis": data.get("fake generate hypothesis", []),
        "model generate hypothesis": data.get("model generate hypothesis", [])
    }
    os.makedirs(saved_path, exist_ok=True)
    output_file_path = os.path.join(saved_path, os.path.basename(file_path).replace("random_", "ranking_res_"))
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"[process_file] Results saved to: {output_file_path}")
    except Exception as e:
        print(f"[process_file] Failed to save {output_file_path}: {e}")

def main():
    """Process all JSON files in the directory."""
    try:
        json_files = [os.path.join(OPT_PATH, file) for file in os.listdir(OPT_PATH) if file.startswith("random_") and file.endswith(".json")]
    except Exception as e:
        print(f"[main] Failed to read directory {OPT_PATH}: {e}")
        return

    total_files = len(json_files)
    print(f"[main] Found {total_files} JSON files to process.")
    start_time = time.time()

    for idx, file_path in enumerate(json_files, 1):
        output_file_path = os.path.join(SAVED_PATH, os.path.basename(file_path).replace("random_", "ranking_res_"))
        if os.path.exists(output_file_path):
            print(f"[main] {output_file_path} already exists, skipping {file_path}")
            continue

        print(f"\n[main] Processing {idx}/{total_files} ({(idx/total_files)*100:.2f}%): {file_path}")
        process_file(file_path)

        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_files - idx)
        print(f"[main] Processed {idx} files, estimated remaining time: {remaining:.1f} seconds")

    print("[main] All files processed!")

if __name__ == "__main__":
    main()
