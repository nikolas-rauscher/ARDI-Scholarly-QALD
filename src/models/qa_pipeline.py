import sys
# Append source directories to system path
sys.path.append('./src')
sys.path.append('..')
import gc  # Garbage Collector Interface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from features.evidence_selection import evidence_triple_selection, triple2text, evidence_sentence_selection
from models.verbalizer.generatePrompt import verbalise_triples
from src.models.zero_shot_prompting_pipeline import clean_context, truncate_context_to_max_chars
import os
from tqdm import tqdm
import configparser
import json

# Read configuration file
config = configparser.ConfigParser()
config.read('config.ini')
max_length_input = config.getint('Parameters', 'max_length_input')
max_output_length = config.getint('Parameters', 'max_output_length')

# Approximate characters per token for model input
chars_per_token = 4
max_length_chars = max_length_input * chars_per_token

# Determine script directory and project prefix path
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)
                                       ).split("/")[:-2]) + "/"
print(PREFIX_PATH)

# Re-read configuration for PREFIX_PATH adjustments
config.read(PREFIX_PATH + 'config.ini')


def qa(model, tokenizer, example, prompt_template):
    """
    Generate prediction using a pre-trained model.

    Args:
        model (transformers.PreTrainedModel): The pre-trained model for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        example (dict): Dictionary containing the question and associated context.
        prompt_template (str): Template string to format the input for the model.

    Returns:
        str: The generated response text.
    """
    context = clean_context(example["contexts"])
    truncated_context = truncate_context_to_max_chars(
        context, example['question'], prompt_template)
    formatted_prompt = prompt_template.format(
        question=example['question'], context=truncated_context)

    inputs = tokenizer(formatted_prompt, add_special_tokens=True,
                       max_length=max_length_input, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_new_tokens=max_output_length)
    response_text = tokenizer.batch_decode(
        outputs, skip_special_tokens=True)[0]
    return response_text

def question_answering(model, tokenizer, prompt_template, question, verbalizer=True, evidence_matching=True):
    """
    Perform question answering with context and evidence processing.

    Args:
        model (transformers.PreTrainedModel): The pre-trained model for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        prompt_template (str): Template string for the input prompt.
        question (str): The question to be answered.
        verbalizer (bool): Whether to use verbalizer for triples. Defaults to True.
        evidence_matching (bool): Whether to match evidence for the answer. Defaults to True.

    Returns:
        str: The generated response based on the processed context.
    """
    wiki_data = {}  # Dictionary to hold Wikipedia data
    all_triples = []  # List to store extracted triples

    # Process Wikipedia data for context
    for key, wiki_text in wiki_data.items():
        if wiki_text and evidence_matching:
            sentences = wiki_text.split('.')
            wiki_evidence = evidence_sentence_selection(
                wiki_data, sentences, conserved_percentage=0.2, max_num=40)
            wiki_context = '. '.join(wiki_evidence)
        else:
            wiki_context = str(wiki_text)

    # Generate context based on options
    if not evidence_matching and not verbalizer:
        context = '. '.join([triple2text(triple)
                            for triple in all_triples if isinstance(triple, dict)])

    if evidence_matching:
        triples_evidence = evidence_triple_selection(question, all_triples)
        if not verbalizer:
            context = '. '.join(
                [triple2text(triple) for triple in triples_evidence if isinstance(triple, dict)])

    if verbalizer and evidence_matching:
        context = verbalise_triples(triples_evidence)
    context += wiki_context

    return qa(model, tokenizer, {"question": question, "context": context}, prompt_template)
