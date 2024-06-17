import features
from features.evidence_selection import evidence_triple_selection, triple2text
import pandas as pd
from transformers import pipeline
import torch
import json
import sys
sys.path.append('./src')


def load_pipeline(model_path="/Volumes/T7/Backup/models/meta-llama/Meta-Llama-3-8B"):
    return pipeline(
        "text-generation", model=model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )


def formatting_prompts_func(prompt_template, examples, max_length=4096):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt_template context and question
    Returns:
        prompt_texts: formated prompt_templates
    """

    prompt_texts = []
    # Iterate through each example in the batch
    for example in examples:
        # Format each example as a prompt_template-response pair
        prompt_text = prompt_template.format(
            question=example['question'], context=example['context'][:max_length-len(example['question'])-len(prompt_template)])
        prompt_texts.append(prompt_text)
    # Return the list of formatted texts
    return prompt_texts


def zero_shot_prompting(pipeline, examples, prompt_template="", evidence_selection=False, verbalizer=False, max_length=4096):
    """
    Generate answers to questions using a zero-shot learning approach with the provided pipeline.

    Args:
        pipeline (transformers.Pipeline): The text-generation pipeline initialized with a specific model.
        examples (list of dict): A batch of examples, where each example includes a 'context' (list of triples) and a 'question' (str).
        evidence_selection (bool, optional): Flag to determine if evidence selection should be performed. Defaults to False.
    Returns:
        list of str: The generated responses for each input example.
    TODO: add verbalizer
    """
    for example in examples:
        triples = evidence_triple_selection(
            example['question'], example['all_tripples']) if evidence_selection else example['all_tripples']
        if (not verbalizer):
            example['context'] = ''.join(
                [triple2text(triple) for triple in triples])
        else:
            example['context'] = ""

    formatted_prompts = formatting_prompts_func(
        prompt_template, examples, max_length)
    responses = []
    for formatted_prompt in formatted_prompts:
        response = pipeline(formatted_prompt)
        responses.append(response[0]['generated_text'])
    return responses


if __name__ == '__main__':
    llama3 = load_pipeline()
    with open("../../data/raw/prompt_template.txt", "r") as file:
        prompt_template = file.read()
    with open('../../data/processed/test_processed_data.json', 'r') as file:
        examples = json.load(file)
    responses = zero_shot_prompting(llama3, prompt_template)
