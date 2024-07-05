import configparser
import json
from tqdm import tqdm
import sys
import os

sys.path.append('./src')
from features.evidence_selection import evidence_triple_selection, triple2text
from models.verbalizer.generatePrompt import verbalise_triples

config = configparser.ConfigParser()
config.read('config.ini')


def prepare_data(examples, prompt_template, output_file):
    """
    Prepare the data by generating contexts for each example.

    Args:
        examples (list): List of examples.
        prompt_template (str): Prompt template.
        output_file (str): Output file path.

    Returns:
        None
    """
    prepared_data = []
    for example in tqdm(examples, desc="Preparing Data"):
  
        tripples_number = len(example['all_tripples'])

        # Plain Triples
        # context_plain = '. '.join([triple2text(triple)
        #                           for triple in example['all_tripples']])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_tripples'])
        context_evidence = '. '.join(
            [triple2text(triple) for triple in triples_evidence])

        # Verbalizer
        # context_verbalizer = verbalise_triples(example['all_tripples'])

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            # "answer": example["answer"],
            "tripples_number": tripples_number,
            "contexts": {
                # "all_tripples": example['all_tripples'],
                # "plain": context_plain,
                # "verbalizer_on_all_tripples": context_verbalizer,
                "evidence_matching": context_evidence,
                "verbalizer_plus_evidence_matching": context_evidence_verbalizer
            }
        }
        if ("answer" in example):
            prepared_example["answer"] = example["answer"]
        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        # Indent added for better formatting
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)


def process_file(input_file_path, prompt_template_path, output_file_path):
    """
    Process the input file and generate prepared data.

    Args:
        input_file_path (str): Input file path.
        prompt_template_path (str): Prompt template file path.
        output_file_path (str): Output file path.

    Returns:
        bool: True if the process is successful, False otherwise.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        return False
    if not os.path.exists(prompt_template_path):
        print(f"Error: Prompt template file '{prompt_template_path}' not found.")
        return False
    
    with open(input_file_path, 'r') as file:
        examples = json.load(file)
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()
    
    prepare_data(examples, prompt_template, output_file_path)
    return True

if __name__ == '__main__':
    with open(config['FilePaths']['test_data_file'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()
    
    output_file = config['FilePaths']['prepared_data_file']
    prepare_data(examples, prompt_template, output_file)