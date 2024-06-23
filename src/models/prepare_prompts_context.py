import json
import configparser
from tqdm import tqdm
import sys
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
        context_plain = '. '.join([triple2text(triple) for triple in example['all_tripples']])
        
        # Evidence Matching
        triples_evidence = evidence_triple_selection(example['question'], example['all_tripples'])
        context_evidence = '. '.join([triple2text(triple) for triple in triples_evidence])
        
        # Verbalizer
        context_verbalizer = verbalise_triples(example['all_tripples'])
        
        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)
        
        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
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

if __name__ == '__main__':
    with open(config['FilePaths']['test_data_file'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()
    
    output_file = config['FilePaths']['prepared_data_file']
    prepare_data(examples, prompt_template, output_file)
