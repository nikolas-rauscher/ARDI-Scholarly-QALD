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
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def ensure_nltk_resources():
    install("nltk")
    nltk.download('wordnet')
    nltk.download('punkt')

def load_gold_standard_qa(gold_path):
    gold_answers = {}
    with open(gold_path) as json_file:
        data = json.load(json_file)
        for ques in data:
            gold_answers[ques['id']] = ques['answer']
    print(f"Gold answers: loaded {len(data)} questions!")
    return gold_answers

def load_system_answers_qa(system_path):
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
    total_questions = len(gold_answers)
    exact_match_count = 0
    meteor_scores = []

    for ques_id in gold_answers:
        gold_answer = gold_answers[ques_id]
        system_answer = system_answers.get(ques_id, "")

        if gold_answer == system_answer:
            exact_match_count += 1
            meteor_scores.append(1.0)
        else:
            gold_tokens = nltk.word_tokenize(gold_answer)
            system_tokens = nltk.word_tokenize(system_answer)
            gold_str = ' '.join(gold_tokens)
            system_str = ' '.join(system_tokens)
            try:
                meteor_scores.append(meteor_score([gold_str], system_str))
            except Exception as err:
                print(f"Error calculating METEOR score for question {ques_id}: {err}")
                meteor_scores.append(0.0)

    exact_match_accuracy = exact_match_count / total_questions if total_questions > 0 else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return exact_match_accuracy, avg_meteor

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_single_file(gt_path, system_path, output_dir):
    gold_answers_qa = load_gold_standard_qa(gt_path)
    system_answers_qa = load_system_answers_qa(system_path)
    exact_match_accuracy, avg_meteor_qa = evaluate_qa(gold_answers_qa, system_answers_qa)
    print(f"QA Results:\n\tExact Match Accuracy: {round(exact_match_accuracy * 100, 2)}%\n\tMETEOR: {round(avg_meteor_qa, 5)}")

    output_file = os.path.join(output_dir, ('scores_' + os.path.basename(system_path).replace('answer_zero_shot_prompting', 'zero-shot')[:-5] + '.txt'))
    with open(output_file, 'w') as f:
        f.write(f"EM: {exact_match_accuracy:.6f}\n")
        f.write(f"METEOR: {avg_meteor_qa:.6f}\n")

def main():
    ensure_nltk_resources()
    input_dir = 'results/fine_tuning_preds_epoch_results_out'
    gt_path = 'data/processed/processed_data_final500_format.json'
    output_dir = 'results/fine_tuning_preds_epoch_results_out/scores/'
    ensure_folder_exists(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            system_path = os.path.join(input_dir, filename)
            print(f"Processing file:\n\tGround truth: {gt_path}\n\tSystem path: {system_path}")
            process_single_file(gt_path, system_path, output_dir)

if __name__ == '__main__':
    gt_path = 'data/raw/trainingdata.json'
    process_single_file(gt_path, "results/experiments_T5/fine-tuning/model_3/answer.txt", "results/experiments_T5/fine-tuning/model_3")
