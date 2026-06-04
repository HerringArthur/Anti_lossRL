"""
Preprocess GSM8K raw JSONL data to validated JSONL format compatible with
validate_anti_loss.py.

Converts raw JSONL (question, answer) to JSONL with:
  - data_source: "openai/gsm8k"
  - prompt: chat-formatted list of messages
  - reward_model: {"style": "rule", "ground_truth": <extracted_answer>}
  - extra_info: metadata

Usage:
  python preprocess_gsm8k.py \
      --input_dir ./data/grade-school-math/grade_school_math/data \
      --output_dir ./data/gsm8k_processed
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def extract_solution(solution_str: str) -> str:
    """Extract final answer after #### from GSM8K solution string."""
    match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if match is None:
        raise ValueError(f"Cannot extract solution from: {solution_str[-200:]}")
    return match.group(1).replace(",", "")


def process_split(input_path: str, output_path: str, split: str):
    """Process a single JSONL file, writing validated JSONL."""
    INSTRUCTION = (
        "Let's think step by step and output the final answer after \"####\"."
    )

    df = pd.read_json(input_path, lines=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            question_raw = row["question"]
            answer_raw = row["answer"]

            question = question_raw + " " + INSTRUCTION
            solution = extract_solution(answer_raw)

            entry = {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": int(idx),
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"[{split}] {count} rows saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GSM8K JSONL → validated JSONL with reward_model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/gsm8k_processed",
        help="Output directory for validated JSONL files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "test"]:
        input_path = os.path.join(args.input_dir, f"{split}.jsonl")
        output_path = os.path.join(args.output_dir, f"{split}.jsonl")
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping {split}")
            continue
        process_split(input_path, output_path, split)


if __name__ == "__main__":
    main()
