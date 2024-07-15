# from https://github.com/debayan/scholarly-QALD-challenge/tree/main

import argparse
import json
import math
import sys
import os
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("nltk")
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')

# Initialize WordNet
nltk.download('punkt')  # Ensure you have punkt tokenizer downloaded


def load_gold_standard_qa(gold_path):
    gold_answers = dict()
    with open(gold_path) as json_file:
        data = json.load(json_file)
    for ques in data:
        gold_answers[ques['id']] = ques['answer']
    print(f"\tGold answers: loaded {len(data)} questions!")
    return gold_answers


def load_system_answers_qa(system_path):
    system_answers = dict()
    with open(system_path) as json_file:
        data = json.load(json_file)
    for ques in data:
        if 'answer' in ques:
            system_answers[ques['id']] = ques['answer']
        else:
            print(f"Missing questions: {ques['id']}")
    print(f"\tSystem answers: loaded {len(data)} questions!")
    return system_answers


def evaluate_qa(gold_answers, system_answers):
    total_questions = len(gold_answers)
    exact_match_count = 0
    meteor_scores = []

    for ques_id in gold_answers:
        gold_answer = gold_answers[ques_id]
        system_answer = system_answers.get(ques_id, "")

        # Calculate exact match
        if gold_answer == system_answer:
            exact_match_count += 1
            meteor_scores.append(1.0)  # Explicitly set METEOR score to 1.0 for exact matches
        else:
            # Tokenize the answers
            gold_tokens = nltk.word_tokenize(gold_answer)
            system_tokens = nltk.word_tokenize(system_answer)

            # Join tokens back into strings for METEOR
            gold_str = ' '.join(gold_tokens)
            system_str = ' '.join(system_tokens)

            # Calculate METEOR for each answer
            try:
                meteor_scores.append(meteor_score([gold_str], system_str))
            except Exception as err:
                print(err)
                meteor_scores.append(0.0)

    # Calculate exact match accuracy
    exact_match_accuracy = exact_match_count / total_questions if total_questions > 0 else 0.0

    # Calculate METEOR score
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return exact_match_accuracy, avg_meteor


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    gt_path = os.path.join(os.path.join(input_dir, 'ref'), 'answer.txt')
    system_path = os.path.join(os.path.join(input_dir, 'res'), 'answer.txt')
    print(f"Config:\n\tGround truth: {gt_path}\n\tSystem path: {system_path}")

    gold_answers_qa = load_gold_standard_qa(gt_path)
    system_answers_qa = load_system_answers_qa(system_path)
    exact_match_accuracy, avg_meteor_qa = evaluate_qa(gold_answers_qa, system_answers_qa)
    print(
        f"\nQA Results:\n\tExact Match Accuracy: {round(exact_match_accuracy * 100, 2)}%\n\tMETEOR: {round(avg_meteor_qa, 5)}")

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        f.write(f"EM: %f\n" % (exact_match_accuracy))
        f.write(f"METEOR: %f\n" % (avg_meteor_qa))
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == '__main__':
    directory = 'results/fine_tuning_preds_epoch_results_out'
    gt_path = 'data/processed/processed_data_final500_format.json'
    output_dir = 'results/fine_tuning_preds_epoch_results_out/scores//' 
    ensure_folder_exists(output_dir)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            system_path = os.path.join(directory, filename)
            print(f"Config:\n\tGround truth: {gt_path}\n\tSystem path: {system_path}")

            gold_answers_qa = load_gold_standard_qa(gt_path)
            system_answers_qa = load_system_answers_qa(system_path)
            exact_match_accuracy, avg_meteor_qa = evaluate_qa(gold_answers_qa, system_answers_qa)
            print(
                f"\nQA Results:\n\tExact Match Accuracy: {round(exact_match_accuracy * 100, 2)}%\n\tMETEOR: {round(avg_meteor_qa, 5)}")

            with open(os.path.join(output_dir, ('scores'+filename[:-5]+'.txt').replace('answer_zero_shot_prompting','_zero-shot')), 'w') as f:
                f.write(f"EM: %f\n" % (exact_match_accuracy))
                f.write(f"METEOR: %f\n" % (avg_meteor_qa))