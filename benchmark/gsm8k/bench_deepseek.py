import argparse
import ast
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np
from tqdm import tqdm

from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999

def get_one_example(lines, i, include_answer):
    """Get one example from the dataset"""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret

def get_few_shot_examples(lines, k):
    """Get k few-shot examples from the dataset"""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret

def get_answer_value(answer_str):
    """Extract the numerical answer from the answer string"""
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", answer_str)
    # Extract numbers from boxed content
    for boxed_content in boxed_matches:
        numbers = re.findall(r"\d+", boxed_content)
        if len(numbers) < 1:
            return INVALID
        return ast.literal_eval(numbers[-1])
    return INVALID

def get_dataset_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID

def call_deepseek_generate(prompt, temperature=0, max_tokens=256, stop=None):
    """
    Call DeepSeek model to generate text by writing to a file and running the generate.py script
    
    Args:
        prompt: The input prompt
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        stop: Stop tokens (not used in this implementation)
        
    Returns:
        The generated text
    """
    # Write the prompt to a file
    with open("question.txt", "w") as f:
        f.write(prompt)
    
    # Run the generate.py script
    cmd = [
        "torchrun", "--nnodes", "1", "--nproc-per-node", "4",
        "/home/aiscuser/yangwang/DeepSeek-V3-inference/generate.py",
        "--ckpt-path", "/home/aiscuser/yangwang/deepseek-r1_reshard_packed/v_8_k_65536_mp4/",
        "--config", "/home/aiscuser/yangwang/DeepSeek-V3/configs/config_671B.json",
        "--quantize",
        "--quant-config", "/home/aiscuser/yangwang/deepseek-r1_reshard_packed/v_8_k_65536_mp4/config.json",
        "--input-file", "/home/aiscuser/yangwang/sglang/benchmark/gsm8k/question.txt",
        "--max-new-tokens", str(max_tokens),
        "--temperature", str(temperature)
    ]
    
    subprocess.run(cmd, check=True)
    
    # Read the output from the file
    with open("output.txt", "r") as f:
        lines = f.readlines()
    
    # Extract the completion part
    for i, line in enumerate(lines):
        if line.startswith("Completion:"):
            return line[len("Completion:"):].strip()
    
    return ""

def main(args):
    # Read data
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    filename = download_and_cache_file(url)
    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    # shuffle the lines
    random.shuffle(lines)
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_dataset_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    states = [None] * len(labels)

    # Clear the output file before starting
    with open("output.txt", "w") as f:
        pass

    # Run requests sequentially (parallel execution would require more complex handling)
    def get_one_answer(i):
        answer = call_deepseek_generate(
            prompt=few_shot_examples + questions[i],
            temperature=0.2,
            max_tokens=16384,
        )
        states[i] = answer

    tic = time.time()
    for i in tqdm(range(len(questions))):
        get_one_answer(i)
    latency = time.time() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")

    # Dump results
    dump_state_text("tmp_output_deepseek.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": "deepseek",
            "num_gpus": 4,  # Using 4 GPUs as specified in the torchrun command
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "model": "DeepSeek-R1-671B",
            },
        }
        fout.write(json.dumps(value) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--result-file", type=str, default="benchmark_results.jsonl")
    args = parser.parse_args()
    main(args) 