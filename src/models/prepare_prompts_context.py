import os
from tqdm import tqdm
import configparser
import sys
sys.path.append('./src')
sys.path.append('..')
import json
from models.verbalizer.generatePrompt import verbalise_triples
from features.evidence_selection import evidence_triple_selection, triple2text, evidence_sentence_selection

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)
                                       ).split("/")[:-2]) + "/"
print(PREFIX_PATH)

config = configparser.ConfigParser()
config.read(PREFIX_PATH + 'config.ini')
config = configparser.ConfigParser()
config.read('config.ini')


def prepare_data_only_ve(examples, prompt_template, output_file):
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
            example['question'], example['all_triples']
        )

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        wiki_context = ""
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
            "contexts": context_evidence_verbalizer + wiki_context
        }

        if "answer" in example:
            prepared_example["answer"] = example["answer"]

        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)





def prepare_data_4settings(examples, prompt_template, output_file, wiki=True):
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
    for example in tqdm(examples, desc="Preparing Data"):

        triples_number = len(example['all_triples'])

        # wiki
        wiki_context = ""
        wiki_context_plain = ""
        if (wiki):
            for wiki_text in example['wiki_data']:
                # sentences = ['. '.join(list(item.values())) for item in wiki_text]
                wiki_context_plain += str(wiki_text)
                if len(wiki_text) > 0:
                    sentences = str(wiki_text).split('.')
                    wiki_evidence = evidence_sentence_selection(
                        example['question'], sentences, conserved_percentage=0.2, max_num=40
                    )
                    wiki_context += '. '.join(wiki_evidence)

        # Plain Triples
        context_plain = '. '.join([triple2text(triple)
                                  for triple in example['all_triples']])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples'][0])
        context_evidence = '. '.join(
            [triple2text(triple) for triple in triples_evidence])

        # Verbalizer
        context_verbalizer = verbalise_triples(example['all_triples'])

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": {
                "all_triples": example['all_triples'],
                "plain": context_plain+wiki_context_plain,
                "verbalizer_on_all_triples": context_verbalizer+wiki_context_plain,
                "evidence_matching": context_evidence+wiki_context,
                "verbalizer_plus_evidence_matching": context_evidence_verbalizer+wiki_context
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
        print(
            f"Error: Prompt template file '{prompt_template_path}' not found.")
        return False

    with open(input_file_path, 'r') as file:
        examples = json.load(file)
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    prepare_data_only_ve(examples, prompt_template, output_file_path)
    return True


if __name__ == '__main__':
    with open("data/processed/train.json", 'r') as file:
        examples = json.load(file)[:100]
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()

    output_file = config['FilePaths']['prepared_data_file']
    prepare_data_only_ve(examples, prompt_template, output_file)
