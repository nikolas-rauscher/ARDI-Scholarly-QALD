import features
from data.make_prompt import formatting_prompts_func
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


def zero_shot_prompting(pipeline, examples, evidence_selection=False):
    """
    Generate answers to questions using a zero-shot learning approach with the provided pipeline.

    Args:
        pipeline (transformers.Pipeline): The text-generation pipeline initialized with a specific model.
        examples (list of dict): A batch of examples, where each example includes a 'context' (list of triples) and a 'question' (str).
        evidence_selection (bool, optional): Flag to determine if evidence selection should be performed. Defaults to False.
    Returns:
        list of str: The generated responses for each input example.
    TODO: add verbaliser
    """
    for example in examples:
        triples = evidence_triple_selection(
            example['question'], example['all_tripples']) if evidence_selection else example['all_tripples']
        example['context'] = ''.join(
            [triple2text(triple) for triple in triples])

    formatted_prompts = formatting_prompts_func(prompt_template, examples)
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
