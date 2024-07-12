import json
import configparser
from tqdm import tqdm
import sys
import os

sys.path.append('./src')
from features.evidence_selection import evidence_triple_selection, triple2text,evidence_sentence_selection
from models.verbalizer.generatePrompt import verbalise_triples

config = configparser.ConfigParser()
config.read('config.ini')

def prepare_data_only_ve(examples, prompt_template, output_file):
    prepared_data = []
    for example in tqdm(examples, desc="Preparing Data"):
        # Anzahl der Tripel
        triples_number = len(example['all_triples'])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples'])

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)
        
        wiki_context=""
        for wiki_text in example['wiki_data']:
            sentences=str(wiki_text)
            wiki_evidence=evidence_sentence_selection(example['question'], sentences, conserved_percentage=0.1, max_num=40)
            wiki_context=wiki_evidence

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": context_evidence_verbalizer+wiki_context
        }
        if ("answer" in example):
            prepared_example["answer"] = example["answer"]
        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        # Indent added for better formatting
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)


def prepare_data(examples, prompt_template, output_file):
    prepared_data = []
    for example in tqdm(examples, desc="Preparing Data"):
        # Anzahl der Tripel
        triples_number = len(example['all_triples'])

        # Plain Triples
        context_plain = '. '.join([triple2text(triple)
                                  for triple in example['all_triples']])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples'])
        context_evidence = '. '.join(
            [triple2text(triple) for triple in triples_evidence])

        # Verbalizer
        context_verbalizer = verbalise_triples(example['all_triples'])

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            # "answer": example["answer"],
            "triples_number": triples_number,
            "contexts": {
                "all_triples": example['all_triples'],
                "plain": context_plain,
                "verbalizer_on_all_triples": context_verbalizer,
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
    
    prepare_data_only_ve(examples, prompt_template, output_file_path)
    return True

if __name__ == '__main__':
    input_files = [
        "./data/processed/processed_data_final500_format.json",
    ]
    
    output_files = [
        './results/prepared_data_final500.json',
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