from rouge_score import rouge_scorer
import os
import json
import numpy as np

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


def calculate_rouge(answers_to_evaluate, reference_answers):
    """
    Calculate ROUGE scores for a set of answers against reference answers.

    Args:
        answers_to_evaluate (list): List of answers to evaluate.
        reference_answers (list): List of reference answers.

    Returns:
        dict: ROUGE scores including precision, recall, and f-measure.
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {}

    for generated_answer in answers_to_evaluate:
        answer_id = generated_answer["id"]
        answer_text = generated_answer["answer"]
        reference = search_dict_list_by_id(reference_answers, answer_id)
        if reference:
            scores = scorer.score(reference["answer"], answer_text)
            all_scores[answer_id] = {k: {'precision': v.precision, 'recall': v.recall, 'fmeasure': v.fmeasure}
                                     for k, v in scores.items()}
        else:
            print(f"No reference answer found for ID {answer_id}")
    return all_scores


def calculate_rouge2(answers_to_evaluate, reference_answers):
    """
    Calculate ROUGE scores for a set of answers against reference answers without IDs.

    Args:
        answers_to_evaluate (list): List of answers to evaluate.
        reference_answers (list): List of reference answers.

    Returns:
        dict: ROUGE scores including precision, recall, and f-measure.
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {}
    for idx, (generated_answer, reference_answer) in enumerate(zip(answers_to_evaluate, reference_answers)):
        scores = scorer.score(reference_answer["answer"], generated_answer)
        all_scores[idx] = {k: {'precision': v.precision, 'recall': v.recall, 'fmeasure': v.fmeasure}
                           for k, v in scores.items()}
    return all_scores


def ensure_folder_exists(folder_path):
    """
    Ensure that a folder exists; create it if it does not.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    directory = '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/processed/100q/processed'
    ref_directory = 'results/train_test_data'
    output_dir = os.path.join(directory, 'evaluation')
    # directory = 'results/hm_results/processed/'
    gt_path = '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/raw/raw_train_dataset.json'
    # output_dir = 'results/hm_results/scores_rouge/'

    score_filename = 'rouge_scores.json'
    ensure_folder_exists(output_dir)

    with open(gt_path, 'r') as f:
        reference_answers = json.load(f)

    for filename in os.listdir(directory):
        if ("answer" not in filename):
            continue
        if filename.endswith(".json"):
            system_path = os.path.join(directory, filename)

        with open(system_path, 'r') as f:
            data = json.load(f)
            all_scores = calculate_rouge(data, reference_answers)
        # Write the scores to a file in JSON format.
        with open(os.path.join(output_dir, ('scores_'+filename[:-5]+'.json').replace('answer_zero_shot_prompting', '_zero-shot')), 'w') as score_file:
            json.dump(all_scores, score_file, indent=4)
            print(f"ROUGE scores saved")

        # Collect statistics
        statistics = {
            'average': {}, 'max': {}, 'median': {}
        }

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

        overview_file_path = os.path.join(
            output_dir, f"overview_scores_{filename.replace('answer_zero_shot_prompting', 'zero-shot')[:-5]}.json")
        with open(overview_file_path, 'w') as stats_file:
            json.dump(statistics, stats_file, indent=2)
