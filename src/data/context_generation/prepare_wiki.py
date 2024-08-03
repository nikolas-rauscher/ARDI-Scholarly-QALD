import os
from tqdm import tqdm
import sys
sys.path.append('./src')
sys.path.append('..')
import configparser
import json
from src.data.verbalizer.prompt_verbalizer import verbalise_triples
from data.evidence_selection.evidence_selection import evidence_triple_selection, triple2text, evidence_sentence_selection

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


def prepare_data_wiki(examples,  output_file):
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

        for wiki_text in example['wiki_data']:
            # sentences = ['. '.join(list(item.values())) for item in wiki_text]
            wiki_context_plain += str(wiki_text)
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
            "contexts": {
                "wiki_context": wiki_context,
                "wiki_context_plain": wiki_context_plain
            }
        }

        if ("answer" in example):
            prepared_example["answer"] = example["answer"]
        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        # Indent added for better formatting
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)
        
        
if __name__ == '__main__':
    with open("./data/processed/wiki.json", 'r') as file:
        examples = json.load(file)

    output_file = config['FilePaths']['prepared_data_file']
    prepare_data_wiki(examples[:5], output_file)