import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.evidence_selection.evidence_selection import evidence_triple_selection, load_triplet_extractor, triple2text, evidence_sentence_selection
from src.data.verbalizer.prompt_verbalizer import verbalise_triples
import json
import configparser
from tqdm import tqdm



config = configparser.ConfigParser()
config.read('config.ini')

def generate_contexts_with_evidence_and_verbalizer(examples, prompt_template, output_file, wikipedia_data=True):
    """
    Prepare the data by generating contexts for each example with all 3 resources

    Args:
        examples (list): List of examples.
        prompt_template (str): Prompt template.
        output_file (str): Output file path.

    Returns:
        None
    """
    prepared_data = []

    for example in tqdm(examples, desc="Preparing Data"):
        # Number of triples
        triples_number = len(example['all_triples'])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples'])
        

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        wiki_context = ""
        if wikipedia_data and 'wiki_data' in example:
            for wiki_text in example['wiki_data']:
                # sentences = ['. '.join(list(item.values())) for item in wiki_text]
                if len(wiki_text) > 0:
                    sentences = str(wiki_text).split('.')
                    wiki_evidence = evidence_sentence_selection(
                        example['question'], sentences, conserved_percentage=0.2, max_num=40
                    )
                    wiki_context += '. '.join(wiki_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": str(context_evidence_verbalizer) + wiki_context
        }

        if "answer" in example:
            prepared_example["answer"] = example["answer"]

        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)

def prepare_data_4settings(examples, prompt_template, output_file, wikipedia_data=True):
    """
    Prepare the data by generating contexts for each example with dnlp and openalex

    Args:
        examples (list): List of examples.
        prompt_template (str): Prompt template.
        output_file (str): Output file path.

    Returns:
        None
    """
    prepared_data = []

    # Process only the first 100 examples
    examples = examples[:100]
    triple_extractor = load_triplet_extractor()

    for example in tqdm(examples, desc="Preparing Data"):

        # Check if all_triples is a list of lists, a flat list, or a dictionary
        if isinstance(example['all_triples'], dict):
            all_triples_flat = [example['all_triples']]
        elif all(isinstance(i, list) for i in example['all_triples']):
            all_triples_flat = [triple for triples in example['all_triples'] for triple in triples]
        else:
            all_triples_flat = example['all_triples']

        triples_number = len(all_triples_flat)

        # wiki
        wiki_context = ""
        wiki_context_plain = ""
        if wikipedia_data and 'wiki_data' in example:
            for wiki_text in example['wiki_data']:
                wiki_context_plain += str(wiki_text)
                if len(wiki_text) > 0:
                    sentences = str(wiki_text).split('.')
                    wiki_evidence = evidence_sentence_selection(
                        example['question'], sentences, conserved_percentage=0.2, max_num=40, llm=True, triplet_extractor=triple_extractor
                    )
                    wiki_context += '. '.join(wiki_evidence)

        # Plain Triples
        context_plain = '. '.join([triple2text(triple) for triple in all_triples_flat if isinstance(triple, dict)])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], all_triples_flat, llm=True, triplet_extractor=triple_extractor)
        context_evidence = '. '.join(
            [triple2text(triple) for triple in triples_evidence if isinstance(triple, dict)])

        # Verbalizer
        context_verbalizer = verbalise_triples(all_triples_flat)

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": {
                "all_triples": all_triples_flat,
                "plain": context_plain + wiki_context_plain,
                "verbalizer_on_all_triples": context_verbalizer + wiki_context_plain,
                "evidence_matching": context_evidence + wiki_context,
                "verbalizer_plus_evidence_matching": context_evidence_verbalizer + wiki_context
            }
        }

        if "answer" in example:
            prepared_example["answer"] = example["answer"]
        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)


def generate_context_for_file(input_file_path, prompt_template_path, output_file_path):
    """
    Generate context for the input file and prepare data.
    It runns the prepare_data_4settings function. So it will generate context for all 4 experiments.

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
        print(
            f"Error: Prompt template file '{prompt_template_path}' not found.")
        return False

    with open(input_file_path, 'r') as file:
        examples = json.load(file)
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    prepare_data_4settings(examples, prompt_template, output_file_path)
    return True


if __name__ == '__main__':
    with open(config['FilePaths']['prepare_prompt_context_input'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()

    output_file = config['FilePaths']['prepared_data_file']
    # generate_contexts_with_evidence_and_verbalizer(examples, prompt_template, output_file, wikipedia_data=True)
    prepare_data_4settings(examples, prompt_template, output_file, wikipedia_data=True)
    
    

    