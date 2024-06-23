import json
import configparser
from tqdm import tqdm
import sys
import os

sys.path.append('./src')
from features.evidence_selection import evidence_triple_selection, triple2text
from models.verbalizer.generatePrompt import verbalise_triples

config = configparser.ConfigParser()
config.read('config.ini')

def prepare_data(examples, prompt_template, output_file):
    prepared_data = []
    for example in tqdm(examples, desc="Preparing Data"):
        # Anzahl der Tripel
        tripples_number = len(example['all_tripples'])
        
        # Plain Triples
        context_plain = ''.join([triple2text(triple) for triple in example['all_tripples']])
        
        # Evidence Matching
        triples_evidence = evidence_triple_selection(example['question'], example['all_tripples'])
        context_evidence = ''.join([triple2text(triple) for triple in triples_evidence])
        
        # Verbalizer
        context_verbalizer = verbalise_triples(example['all_tripples'])
        
        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)
        
        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            # "answer": example["answer"],
            "tripples_number": tripples_number,
            "contexts": {
                "all_tripples": example['all_tripples'],
                "plain": context_plain,
                "verbalizer_on_all_tripples": context_verbalizer,
                "evidence_matching": context_evidence,
                "verbalizer_plus_evidence_matching": context_evidence_verbalizer
            }
        }
        prepared_data.append(prepared_example)
    
    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)  # Indent added for better formatting

def process_file(input_file_path, prompt_template_path, output_file_path):
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
    input_files = [
        'data/external/train_post_processed_data_dblp_hm.json',
        'data/external/train_post_processed_data_alex_hm.json',
        'data/external/train_hm_openalex_dblp.json'
    ]
    
    output_files = [
        './results/prepared_data_hm_openalex_dblp.json',
        './results/prepared_data_alex_hm.json',
        './results/prepared_data_dblp_hm.json'
    ]
    
    prompt_template_path = config['FilePaths']['prompt_template']

    all_files_exist = all(os.path.exists(f) for f in input_files + [prompt_template_path])
    if not all_files_exist:
        print("Error: One or more input files or the prompt template file are missing.")
        sys.exit(1)

    for input_file, output_file in zip(input_files, output_files):
        success = process_file(input_file, prompt_template_path, output_file)
        if not success:
            print(f"Processing stopped due to missing file: {input_file}")
            break