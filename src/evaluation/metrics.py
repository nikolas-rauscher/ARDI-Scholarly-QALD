from nltk.translate.meteor_score import meteor_score
import nltk
from sklearn.metrics import precision_recall_fscore_support
from rouge_score import rouge_scorer
import evaluate
import os
import json
import numpy as np
metric = evaluate.load("rouge")

def calculate_metrics(generated_answers, reference_answers):
    """
    Calculate ROUGE scores for a set of answers against reference answers with corresponding order

    Args:
        generated_answers (list): List of generated answers to evaluate.
        reference_answers (list): List of reference answers.

    Returns:
        dict: ROUGE scores including precision, recall, and f-measure.
    """
    reference_answers = generate_reference_answers_from_ids(
        generated_answers, reference_answers)
    return compute_metrics([item["answer"] for item in generated_answers], [item["answer"] for item in reference_answers])

# ==================================================================================
# ==============Functions basically never called externally=========================
# ==================================================================================


def compute_meteor(generated_answers, reference_answers):
    total_questions = len(generated_answers)
    exact_match_count = 0
    meteor_scores = []
    for system_answer, gold_answer in zip(generated_answers, reference_answers):
        if gold_answer == system_answer:
            exact_match_count += 1
            meteor_scores.append(1.0)
        else:
            gold_tokens = nltk.word_tokenize(gold_answer)
            system_tokens = nltk.word_tokenize(system_answer)
            try:
                meteor_scores.append(meteor_score([system_tokens], gold_tokens))
            except Exception as err:
                print(f"Error calculating METEOR score : {err}")
                meteor_scores.append(0.0)

    exact_match_accuracy = exact_match_count / \
        total_questions if total_questions > 0 else 0.0
    avg_meteor = sum(meteor_scores) / \
        len(meteor_scores) if meteor_scores else 0.0

    return exact_match_accuracy, avg_meteor


def ensure_nltk_resources():
    """Ensures necessary NLTK resources are downloaded."""
    nltk.download('wordnet')
    nltk.download('punkt')

def compute_metrics(preds, labels):
    if isinstance(preds, tuple):
        preds = preds[0]

    grounds, preds = labels, preds
    p, r, f, s = precision_recall_fscore_support(
        grounds, preds, labels=labels, average='micro')

    # Compute ROUGscores
    result = metric.compute(
        predictions=preds, references=labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(len(preds))
    result['f1'] = f
    result['recall'] = r
    result['precision'] = p
    exact_match_accuracy, avg_meteor = compute_meteor(preds, labels)
    result['EM'] = exact_match_accuracy
    result['meteor'] = avg_meteor

    return {k: round(v, 4) for k, v in result.items()}


def search_dict_list_by_id(dict_list, id_idx):
    """
    Search for an item in a list of dictionaries by the 'id' key.

    Args:
        dict_list (list): List of dictionaries.
        id_idx (str): The id to search for.

    Returns:
        dict: The found dictionary or None if not found.
    """
    for item in dict_list:
        if item.get('id') == id_idx:
            return item
    return None


def generate_reference_answers_from_ids(generated_answers, unordered_reference_answers):
    reference_answers = []
    for generated_answer in generated_answers:
        reference_answer = search_dict_list_by_id(
            unordered_reference_answers, generated_answer["id"])
        reference_answers.append(reference_answer)
    return reference_answers


def score_answers(reference_answer, generated_answer, scorer):
    """
    Score a generated answer against a reference answer using ROUGE.

    Args:
        reference_answer (str): The reference answer.
        generated_answer (str): The generated answer.
        scorer (RougeScorer): The ROUGE scorer object.

    Returns:
        dict: ROUGE scores including precision, recall, and f-measure.
    """
    scores = scorer.score(reference_answer, generated_answer)
    return {k: {'precision': v.precision, 'recall': v.recall, 'fmeasure': v.fmeasure}
            for k, v in scores.items()}


def calculate_rouge(generated_answers, reference_answers):
    """
    Calculate ROUGE scores for a set of answers against reference answers with corresponding order

    Args:
        generated_answers (list): List of generated answers to evaluate.
        reference_answers (list): List of reference answers.

    Returns:
        dict: ROUGE scores including precision, recall, and f-measure.
    """
    reference_answers = generate_reference_answers_from_ids(
        generated_answers, reference_answers)
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {}
    for idx, (generated_answer, reference_answer) in enumerate(zip(generated_answers, reference_answers)):
        all_scores[idx] = score_answers(
            reference_answer["answer"], generated_answer, scorer)
    return all_scores


def ensure_folder_exists(directory_path):
    """
    Ensure that a folder exists; create it if it does not.

    Args:
        directory_path (str): The path to the folder.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def collect_statistics(all_scores):
    """
    Collect and calculate statistics for ROUGE scores.

    Args:
        all_scores (dict): Dictionary of all ROUGE scores.

    Returns:
        dict: Dictionary containing average, max, and median statistics.
    """
    statistics = {'average': {}, 'max': {}, 'median': {}}
    all_precisions = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}
    all_recalls = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}
    all_fmeasures = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}

    for rouge_types in all_scores.values():
        for rouge_type, scores in rouge_types.items():
            all_precisions[rouge_type].append(scores['precision'])
            all_recalls[rouge_type].append(scores['recall'])
            all_fmeasures[rouge_type].append(scores['fmeasure'])

    for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
        statistics['average'][rouge_type] = {
            'precision': np.mean(all_precisions[rouge_type]),
            'recall': np.mean(all_recalls[rouge_type]),
            'fmeasure': np.mean(all_fmeasures[rouge_type]),
        }
        statistics['max'][rouge_type] = {
            'precision': np.max(all_precisions[rouge_type]),
            'recall': np.max(all_recalls[rouge_type]),
            'fmeasure': np.max(all_fmeasures[rouge_type]),
        }
        statistics['median'][rouge_type] = {
            'precision': np.median(all_precisions[rouge_type]),
            'recall': np.median(all_recalls[rouge_type]),
            'fmeasure': np.median(all_fmeasures[rouge_type]),
        }

    return statistics


def process_rouge_for_all_files(directory, gt_path):
    """
    Main function to process ROUGE scoring and statistics for answer evaluation.
    """
    output_dir = os.path.join(directory, 'evaluation')

    ensure_folder_exists(output_dir)

    with open(gt_path, 'r') as f:
        reference_answers = json.load(f)

    for filename in os.listdir(directory):
        if "answer" not in filename or not filename.endswith(".json"):
            continue

        answer_file_path = os.path.join(directory, filename)

        with open(answer_file_path, 'r') as f:
            generated_answers = json.load(f)
            all_scores = calculate_rouge(generated_answers, reference_answers)

        # Write the scores to a file in JSON format.
        score_file_path = os.path.join(
            output_dir, ('scores_'+filename[:-5]+'.json').replace('answer_zero_shot_prompting', '_zero-shot'))
        with open(score_file_path, 'w') as score_file:
            json.dump(all_scores, score_file, indent=4)
            print(f"ROUGE scores saved to {score_file_path}")

        # Collect and write statistics.
        statistics = collect_statistics(all_scores)
        overview_file_path = os.path.join(
            output_dir, f"overview_scores_{filename.replace('answer_zero_shot_prompting', 'zero-shot')[:-5]}.json")
        with open(overview_file_path, 'w') as stats_file:
            json.dump(statistics, stats_file, indent=2)
            print(f"Statistics saved to {overview_file_path}")


def process_metric_for_all_files(directory, gt_path):
    """
    Main function to process ROUGE scoring and statistics for answer evaluation.
    """
    output_dir = os.path.join(directory, 'evaluation')

    ensure_folder_exists(output_dir)

    with open(gt_path, 'r') as f:
        reference_answers = json.load(f)

    for filename in os.listdir(directory):
        if "answer" not in filename or not filename.endswith(".json"):
            continue

        answer_file_path = os.path.join(directory, filename)

        with open(answer_file_path, 'r') as f:
            generated_answers = json.load(f)
            all_scores = calculate_metrics(
                generated_answers, reference_answers)

        # Write the scores to a file in JSON format.
        score_file_path = os.path.join(
            output_dir, ('metrics_'+filename[:-5]+'.json'))
        with open(score_file_path, 'w') as score_file:
            json.dump(all_scores, score_file, indent=4)
            print(f"Metrics saved to {score_file_path}")


ensure_nltk_resources()
