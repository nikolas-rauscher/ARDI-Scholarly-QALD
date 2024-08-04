import configparser
import json

from data.prepare_prompts_context import prepare_data_4settings

config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == '__main__':
    with open(config['FilePaths']['prepare_prompt_context_input'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()

    output_file = config['FilePaths']['prepared_data_file']
    # generate_contexts_with_evidence_and_verbalizer(examples, prompt_template, output_file, wikipedia_data=True)
    prepare_data_4settings(examples, output_file, wikipedia_data=True)
