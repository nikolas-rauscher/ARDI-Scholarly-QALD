import os
import json

# Define the list of result keys
RESULT_KEYS = ["all_triples_results", "verbalizer_results",
               "evidence_matching", "verbalizer_plus_evidence_matching"]


def search_dict_list_by_answer(dict_list, answer):
    found_item = None
    for item in dict_list:
        if item.get('answer') == answer:
            found_item = item
            break
    return found_item


def add_question_id(data, gt_data):
    for entity in data:
        if ('id' not in entity):
            entity['id'] = search_dict_list_by_answer(
                gt_data, entity['answer'])['id']
    return data


def pipeline_add_question_ids(test_directory, gt_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    for filename in os.listdir(test_directory):
        print(filename)
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(test_directory, filename), 'r') as f:
            data = json.load(f)
        data = add_question_id(data=data, gt_data=gt_data)
        with open(os.path.join(test_directory, filename), 'w') as f:
            json.dump(data, f, indent=2)


def pipeline_question_ids(test_directory, ref_directory, out_directory):
    for filename in os.listdir(test_directory):
        if (not filename.endswith(".json")) or('train' in filename):
            continue
        with open(os.path.join(test_directory, filename), 'r') as f:
            data = json.load(f)
        matching_files = [f for f in os.listdir(
            ref_directory) if f.startswith(filename[:6])]
        with open(os.path.join(ref_directory, matching_files[0]), 'r') as f:
            reference_answers = json.load(f)
        output = []
        for answer, ref_answer in zip(data, reference_answers):
            output.append({'id': ref_answer['id'], 'answer': answer})
        with open(os.path.join(out_directory, filename), 'w') as f:
            json.dump(output, f, indent=2)


def extract_response(response):
    """Extract and clean the response."""
    if len(response) > 300:
        response = response.split("Answer:")[-1].strip()
        if '[INST]' in response:
            response = response.split("INST]")[-1].strip()
        if 'assistant' in response:
            response = response.split("assistant")[-1].strip()
        if len(response) > 300:
            response = response.split("\n")[-1].strip()
        if len(response) > 300:
            response = 'out of tokens'
    return response


def process_for_evaluation(data, result_keys):
    """Process data for evaluation and generate output."""
    output = [[] for _ in range(len(result_keys))]
    for item in data:
        responses = item["results"][0]
        for i, key in enumerate(result_keys):
            response = extract_response(responses[key]['response'])
            output[i].append({'id': item['id'], 'answer': response})
    return output


def process_file(file_path, process_func, output_dir, result_keys):
    """Process a single file and save the output."""
    with open(file_path, "r") as file:
        data = json.load(file)
        output = process_func(data, result_keys)
        for i, result in enumerate(output):
            output_file_path = os.path.join(
                output_dir, f"answer_{os.path.basename(file_path)[:-5]}_{result_keys[i]}.json")
            with open(output_file_path, "w") as wf:
                json.dump(result, wf, indent=2)
            print(f"Processed and saved: {output_file_path}")


def process_all_files_in_folder(directory, process_func, result_keys):
    """Process all JSON files in the given directory."""
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    output_dir = os.path.join(directory, 'processed')
    os.makedirs(output_dir, exist_ok=True)

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        print(f"Processing file: {file_name}")
        process_file(file_path, process_func, output_dir, result_keys)
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == "__main__":
    test_directory = "results/train_test_data"
    test_directory = "results/train_test_data"
    gt_path = "data/processed/processed_data_final500_format.json"
    data_directories = ['results/zero-shot_Flan_T5_large','results/fine_tuning_preds_epoch_results']
    pipeline_add_question_ids(test_directory, gt_path)
    
    for data_directory in data_directories:
        ensure_folder_exists(data_directory+'_out')
        pipeline_question_ids(test_directory=data_directory,
                              ref_directory=test_directory, out_directory=data_directory+'_out')
    # process_all_files_in_folder(
    #     "./results/hm_results", process_for_evaluation, RESULT_KEYS)
