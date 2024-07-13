import os
import json

# Define the list of result keys
RESULT_KEYS = ["all_triples_results", "verbalizer_results",
               "evidence_matching", "verbalizer_plus_evidence_matching"]


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


if __name__ == "__main__":
    process_all_files_in_folder(
        "./results/hm_results", process_for_evaluation, RESULT_KEYS)
