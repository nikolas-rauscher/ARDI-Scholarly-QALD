# based on https://github.com/debayan/scholarly-QALD-challenge/blob/main/2024/dataset/qa_eval.py

import argparse
import json
import math
import sys
import os
import subprocess
import nltk
from nltk.translate.meteor_score import meteor_score


def install(package):
    """Installs a package using pip.

    Args:
        package (str): The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_nltk_resources():
    """Ensures necessary NLTK resources are downloaded."""
    install("nltk")
    nltk.download('wordnet')
    nltk.download('punkt')


def load_gold_standard_qa(gold_path):
    """Loads the gold standard QA data from a JSON file.

    Args:
        gold_path (str): Path to the JSON file containing gold standard answers.

    Returns:
        dict: A dictionary mapping question IDs to their gold standard answers.
    """
    gold_answers = {}
    with open(gold_path) as json_file:
        data = json.load(json_file)
        for ques in data:
            gold_answers[ques['id']] = ques['answer']
    print(f"Gold answers: loaded {len(data)} questions!")
    return gold_answers


def load_system_answers_qa(system_path):
    """Loads system-provided QA answers from a JSON file.

    Args:
        system_path (str): Path to the JSON file containing system answers.

    Returns:
        dict: A dictionary mapping question IDs to their system answers.
    """
    system_answers = {}
    with open(system_path) as json_file:
        data = json.load(json_file)
        for ques in data:
            if 'answer' in ques:
                system_answers[ques['id']] = ques['answer']
            else:
                print(f"Missing question: {ques['id']}")
    print(f"System answers: loaded {len(data)} questions!")
    return system_answers


def evaluate_qa(gold_answers, system_answers):
    """Evaluates the QA performance using exact match and METEOR scores.

    Args:
        gold_answers (dict): Dictionary of gold standard answers.
        system_answers (dict): Dictionary of system answers.

    Returns:
        tuple: (exact_match_accuracy, avg_meteor) where each is a float.
    """
    total_questions = len(gold_answers)
    exact_match_count = 0
    meteor_scores = []

    for ques_id, gold_answer in gold_answers.items():
        system_answer = system_answers.get(ques_id, "")

        if gold_answer == system_answer:
            exact_match_count += 1
            meteor_scores.append(1.0)
        else:
            gold_tokens = nltk.word_tokenize(gold_answer)
            system_tokens = nltk.word_tokenize(system_answer)
            try:
                meteor_scores.append(meteor_score([system_tokens], gold_tokens))
            except Exception as err:
                print(f"Error calculating METEOR score for question {ques_id}: {err}")
                meteor_scores.append(0.0)

    exact_match_accuracy = exact_match_count / total_questions if total_questions > 0 else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return exact_match_accuracy, avg_meteor


def ensure_folder_exists(folder_path):
    """Ensures that the specified folder exists.

    Args:
        folder_path (str): The path to the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def process_single_file(gt_path, system_path, output_dir):
    """Processes a single file, calculating QA evaluation metrics.

    Args:
        gt_path (str): Path to the ground truth file.
        system_path (str): Path to the system answers file.
        output_dir (str): Directory to save the output scores.
    """
    gold_answers_qa = load_gold_standard_qa(gt_path)
    system_answers_qa = load_system_answers_qa(system_path)
    exact_match_accuracy, avg_meteor_qa = evaluate_qa(gold_answers_qa, system_answers_qa)
    print(f"QA Results:\n\tExact Match Accuracy: {round(exact_match_accuracy * 100, 2)}%\n\tMETEOR: {round(avg_meteor_qa, 5)}")

    output_file = os.path.join(output_dir, f"scores_{os.path.basename(system_path).replace('answer_zero_shot_prompting', 'zero-shot')[:-5]}.txt")
    with open(output_file, 'w') as f:
        f.write(f"EM: {exact_match_accuracy:.6f}\n")
        f.write(f"METEOR: {avg_meteor_qa:.6f}\n")


def main():
    """Main function to execute the QA evaluation process."""
    ensure_nltk_resources()

    # Define paths
    input_dir = '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/processed/100q/processed'
    gt_path = '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/raw/raw_train_dataset.json'
    output_dir = '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/processed/100q/processed/score'

    ensure_folder_exists(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            system_path = os.path.join(input_dir, filename)
            print(f"Processing file:\n\tGround truth: {gt_path}\n\tSystem path: {system_path}")
            process_single_file(gt_path, system_path, output_dir)


if __name__ == '__main__':
    main()
