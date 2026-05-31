# ResearchBench

<p align="center">
  <a href="https://arxiv.org/abs/2503.21248"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="https://ankitala.github.io/ResearchBench/"><img src="https://img.shields.io/badge/Project-Page-2ea44f?logo=githubpages" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/ankilok/ResearchBench"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?logo=huggingface" alt="Dataset"></a>
  <a href="https://github.com/ankitala/ResearchBench"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
</p>

[English](README.md) | [中文](README_zh.md)

ResearchBench is the official code release for the ACL Findings 2026 paper [ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](https://arxiv.org/abs/2503.21248).

This GitHub repository contains the evaluation toolkit, command-line interface, prompts, metrics, tests, and a 12-sample `tiny` smoke-test split. The full benchmark dataset is released on [Hugging Face](https://huggingface.co/datasets/ankilok/ResearchBench): [ankilok/ResearchBench](https://huggingface.co/datasets/ankilok/ResearchBench).

ResearchBench evaluates large language models on scientific discovery through three inspiration-based subtasks:

1. **Inspiration retrieval**: select the papers that can inspire a target research hypothesis.
2. **Hypothesis composition**: compose a research hypothesis from a research question and gold inspirations.
3. **Hypothesis ranking**: choose the better hypothesis in pairwise comparisons between a gold hypothesis and negative hypotheses.

The full release contains **1,369 paper records**, **1,367 retrieval tasks**, **1,367 generation tasks**, and **1,323 ranking tasks**.

## 🚀 Quick Start

Clone the code repository and run all three subtasks on the bundled `data/tiny/*.jsonl` smoke-test split with an OpenAI-compatible provider:

```bash
git clone https://github.com/ankitala/ResearchBench.git
cd ResearchBench

python -m pip install -e ".[openai]"

export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<your-base-url>"
MODEL="<model-name>"

researchbench run-retrieve --data data/tiny/retrieve.jsonl --model-name "$MODEL" --concurrency 15 --out outputs/retrieve_tiny.jsonl
researchbench score-retrieve --pred outputs/retrieve_tiny.jsonl --data data/tiny/retrieve.jsonl

researchbench run-generate --data data/tiny/generation.jsonl --model-name "$MODEL" --num-mutations 2 --num-itr-self-refine 1 --concurrency 15 --out outputs/generation_tiny.jsonl
researchbench score-generate --pred outputs/generation_tiny.jsonl --data data/tiny/generation.jsonl --judge-model "$MODEL" --concurrency 15

researchbench run-rank --data data/tiny/ranking.jsonl --model-name "$MODEL" --order both --concurrency 15 --out outputs/ranking_tiny.jsonl
researchbench score-rank --pred outputs/ranking_tiny.jsonl
```

## 📊 Benchmark Results

The figures below summarize model performance on the three ResearchBench subtasks. Higher scores indicate better performance.

### 🔍 Inspiration Retrieval

![Inspiration retrieval results](figure/retrieval.png)

### 💡 Hypothesis Composition

![Hypothesis composition results](figure/composition.png)

### 🏆 Hypothesis Ranking

![Hypothesis ranking results](figure/ranking.png)

## ⚙️ Installation

ResearchBench requires Python 3.10 or later.

Core local tooling:

```bash
python -m pip install -e .
```

Development and tests:

```bash
python -m pip install -e ".[dev]"
```

OpenAI-compatible provider calls:

```bash
python -m pip install -e ".[openai]"
export OPENAI_API_KEY="<provider-api-key>"
export OPENAI_BASE_URL="<provider-base-url>"
```

You can also pass provider settings per command:

```bash
researchbench run-retrieve \
  --data data/tiny/retrieve.jsonl \
  --model-name "<provider-model-name>" \
  --api-key "<provider-api-key>" \
  --base-url "<provider-base-url>" \
  --concurrency 15 \
  --out outputs/retrieve_provider.jsonl
```

`pyproject.toml` is the authoritative package configuration. `requirements.txt` is a convenience file for installing the editable package with development and OpenAI-compatible provider extras.

<a id="data-layout"></a>

## 🗂️ Data Layout

This GitHub repository is code-first. It tracks only the small `tiny` JSONL files needed for smoke tests:

- `data/tiny/papers.jsonl`: 12-sample paper records.
- `data/tiny/retrieve.jsonl`: 12-sample inspiration retrieval tasks.
- `data/tiny/generation.jsonl`: 12-sample hypothesis composition tasks.
- `data/tiny/ranking.jsonl`: 12-sample hypothesis ranking tasks.

The full benchmark files are hosted on [Hugging Face](https://huggingface.co/datasets/ankilok/ResearchBench) and are not uploaded to GitHub:

- `papers/papers.jsonl`: standardized paper records.
- `retrieve/retrieve.jsonl`: inspiration retrieval tasks.
- `generation/generation.jsonl`: hypothesis composition tasks.
- `ranking/ranking.jsonl`: hypothesis ranking tasks.
- `validation/*.csv` and `validation/*.json`: build and validation reports.

Download any subset as needed:

```bash
huggingface-cli download ankilok/ResearchBench \
  --repo-type dataset \
  --include "retrieve/*.jsonl" \
  --include "generation/*.jsonl" \
  --include "ranking/*.jsonl" \
  --local-dir data
```

Validate a local data directory:

```bash
researchbench validate data
```

Validation checks candidate counts, gold-label coverage, empty fields, local absolute paths, API key leakage, and task-specific shape constraints.

## 🛠️ Basic Usage

All `run-*` commands support `--concurrency`, defaulting to 15. `run-generate` also supports `--inner-concurrency` for independent within-sample branches. Use lower values if your provider has strict rate limits.

**Before running the full benchmark, download the corresponding JSONL files from [Hugging Face](https://huggingface.co/datasets/ankilok/ResearchBench) into `data/` as shown in the [Data Layout](#data-layout) section.**

### 🔍 Inspiration Retrieval

```bash
researchbench run-retrieve \
  --data data/retrieve/retrieve.jsonl \
  --model-name "<model>" \
  --concurrency 15 \
  --out outputs/retrieve.jsonl

researchbench score-retrieve \
  --pred outputs/retrieve.jsonl \
  --data data/retrieve/retrieve.jsonl
```

Default settings match the original experiment: window size 15, keep size 3, two rounds, no background survey, no similarity-only selection. Round 1 keeps 15 of 75 candidates; round 2 keeps 3 of 75.

### 💡 Hypothesis Composition

```bash
researchbench run-generate \
  --data data/generation/generation.jsonl \
  --model-name "<model>" \
  --num-mutations 2 \
  --num-itr-self-refine 2 \
  --max-inspiration-search-steps 3 \
  --concurrency 15 \
  --inner-concurrency 1 \
  --out outputs/generation.jsonl

researchbench score-generate \
  --pred outputs/generation.jsonl \
  --data data/generation/generation.jsonl \
  --judge-model "<judge-model>" \
  --concurrency 15
```

Generation uses `main_hypothesis` as the gold hypothesis. `fine_grained_hypothesis` is retained only as optional metadata. The default generation run uses gold inspirations, two mutation lines, two refinement iterations, same-inspiration recombination, cross-inspiration recombination up to three inspiration steps, inter-EA screening window size 12, keep size 3, and recombination beam size 15.

### 🏆 Hypothesis Ranking

```bash
researchbench run-rank \
  --data data/ranking/ranking.jsonl \
  --model-name "<model>" \
  --order both \
  --concurrency 15 \
  --out outputs/ranking.jsonl

researchbench score-rank \
  --pred outputs/ranking.jsonl
```

`--order both` evaluates both comparison directions. You may also run `--order gold-first` or `--order negative-first`。

## 📋 Example CLI Output

Retrieve scoring prints a readable summary:

```text
Retrieve score summary
- Round 1: kept 15 of 75 candidates
  - Gold inspiration hit ratio: 0.7419 (74.19%)
  - Negative tier 1 selection ratio: 0.5463 (54.63%)
  - Negative tier 2 selection ratio: 0.0833 (8.33%)
  - Negative tier 3 selection ratio: 0.0189 (1.89%)
- Round 2: kept 3 of 75 candidates
  - Gold inspiration hit ratio: 0.3548 (35.48%)
```

Generation scoring:

```text
Hypothesis generation score summary
- Scored samples: 12
- Average matched score (0-5): 4.1667
- Generation accuracy (average score / 5): 0.8333 (83.33%)
```

Ranking scoring:

```text
Hypothesis ranking score summary
- Overall directional accuracy: 0.3750 (37.50%)
- Gold-first order accuracy (gold is candidate 1): 0.3333 (33.33%)
- Negative-first order accuracy (gold is candidate 2): 0.4167 (41.67%)
- Sample-level two-order outcomes
  - Gold wins in neither order: 0.5000 (50.00%); count=6
  - Gold wins in exactly one order: 0.2500 (25.00%); count=3
  - Gold wins in both orders: 0.2500 (25.00%); count=3
```

The full machine-readable metric objects are written next to prediction files as `*.score.json` unless `--out` is specified.

## 📊 Metrics

- **Retrieve gold hit ratio**: selected gold inspirations divided by total gold inspirations. Reported for round 1 and round 2.
- **Retrieve negative selection ratio**: selected negative candidates in a distance tier divided by total candidates in that tier. This supports the negative-distance analysis.
- **Generation matched score**: judge score from 0 to 5 measuring how well the generated hypothesis covers the gold hypothesis key points.
- **Generation accuracy**: average matched score divided by 5.
- **Ranking directional accuracy**: whether the gold hypothesis wins under a single candidate order.
- **Ranking sample two-order outcomes**: whether gold wins in neither, exactly one, or both of the two reversed orders.
- **Ranking pair-level order consistency**: whether both orders choose gold, both choose negative, or the two orders disagree.

## 📝 Notes

- Full benchmark JSONL files are released through the [Hugging Face dataset repository](https://huggingface.co/datasets/ankilok/ResearchBench), not through this GitHub repository.
- Retrieve candidates are read from `merge.json`; `d0.json`, `d1.json`, `d2.json`, and `d3.json` label candidates as gold, negative tier 1, negative tier 2, and negative tier 3.
- Keep API keys in environment variables or pass them through CLI arguments.

## ⚖️ License

Code is released under the MIT License. The dataset is released on [Hugging Face](https://huggingface.co/datasets/ankilok/ResearchBench) under CC BY-NC 4.0 for non-commercial research use; see [DATA_LICENSE.md](DATA_LICENSE.md).

## 📑 Citation

If you use ResearchBench, please cite:

```bibtex
@misc{liu2026researchbenchbenchmarkingllmsscientific,
      title={ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition}, 
      author={Yujie Liu and Zonglin Yang and Tong Xie and Jinjie Ni and Ben Gao and Yuqiang Li and Shixiang Tang and Wanli Ouyang and Erik Cambria and Dongzhan Zhou},
      year={2026},
      eprint={2503.21248},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.21248}, 
}
```
