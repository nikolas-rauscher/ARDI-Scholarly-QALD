from rouge_score import rouge_scorer
import os
import json
import numpy as np


def search_dict_list_by_id(dict_list, id_idx):
    found_item = None
    for item in dict_list:
        if item.get('id') == id_idx:
            found_item = item
            break
    return found_item


def calculate_rouge(answers_to_evaluate, reference_answers):
    # Initialize a ROUGE scorer.
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize a dictionary to hold all our scores.
    all_scores = {}

    # Score each answer.
    for generated_answer in answers_to_evaluate:
        answer_id = generated_answer["id"]
        answer_text = generated_answer["answer"]
        item = search_dict_list_by_id(reference_answers, answer_id)
        reference_text = item.get("answer")

        if reference_text:
            scores = scorer.score(reference_text, answer_text)

            # Create a dictionary for each score type with precision, recall, and fmeasure.
            all_scores[answer_id] = {
                score_type: {
                    'precision': score.precision,
                    'recall': score.recall,
                    'fmeasure': score.fmeasure
                } for score_type, score in scores.items()
            }
        else:
            print(f"No reference answer found for ID {answer_id}")
    return all_scores


def calculate_rouge2(answers_to_evaluate, reference_answers):
    # Initialize a ROUGE scorer.
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize a dictionary to hold all our scores.
    all_scores = {}
    idx=0
    # Score each answer.
    for generated_answer,reference_answer in zip(answers_to_evaluate,reference_answers):
        answer_text = generated_answer
        reference_text = reference_answer["answer"]

        if reference_text:
            scores = scorer.score(reference_text, answer_text)

            # Create a dictionary for each score type with precision, recall, and fmeasure.
            all_scores[idx] = {
                score_type: {
                    'precision': score.precision,
                    'recall': score.recall,
                    'fmeasure': score.fmeasure
                } for score_type, score in scores.items()
            }
            idx+=1
        else:
            print(f"No reference answer found for ID {answer_id}")
    return all_scores

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == '__main__':
    directory = 'results/zero-shot_Flan_T5_large'
    gt_path = 'data/processed/processed_data.json'
    ref_directory = 'results/train_test_data'
    output_dir = 'results/zero-shot_Flan_T5_large/evaluation'

    score_filename = 'rouge_scores.json'
    ensure_folder_exists(output_dir)

    # with open(gt_path, 'r') as f:
    #     reference_answers = json.load(f)

    for filename in os.listdir(directory):
        matching_files = [f for f in os.listdir(ref_directory) if f.startswith(filename[:6])]
        if (("test_" not in filename) or (len(matching_files)==0)):
            continue
        if filename.endswith(".json"):
            system_path = os.path.join(directory, filename)
        system_path_ref = os.path.join(ref_directory, matching_files[0])
        with open(system_path_ref, 'r') as f:
            reference_answers=json.load(f)
        with open(system_path, 'r') as f:
            data = json.load(f)
            all_scores = calculate_rouge2(data, reference_answers)
            # all_scores = calculate_rouge(data, reference_answers)
        # Write the scores to a file in JSON format.
        with open(os.path.join(output_dir, ('scores'+filename[:-5]+'.json').replace('answer_zero_shot_prompting', '_zero-shot')), 'w') as score_file:
            json.dump(all_scores, score_file, indent=4)
            print(f"ROUGE scores saved")

        statistics = {
            'average': {},
            'max': {},
            'median': {},
        }

        # Initialize lists to collect all score values for calculation.
        all_precisions = {rouge_type: []
                          for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
        all_recalls = {rouge_type: []
                       for rouge_type in ['rouge1', 'rouge2', 'rougeL']}
        all_fmeasures = {rouge_type: []
                         for rouge_type in ['rouge1', 'rouge2', 'rougeL']}

        # Collect all score values.
        for answer_id, rouge_types in all_scores.items():
            for rouge_type, rouge_scores in rouge_types.items():
                all_precisions[rouge_type].append(rouge_scores['precision'])
                all_recalls[rouge_type].append(rouge_scores['recall'])
                all_fmeasures[rouge_type].append(rouge_scores['fmeasure'])

        # Calculate statistics for all ROUGE types.
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
        with open(os.path.join(output_dir, ('overview_scores'+filename[:-5]+'.json').replace('answer_zero_shot_prompting', '_zero-shot')), 'w')  as stats_file:
            json.dump(statistics, stats_file, indent=2)
