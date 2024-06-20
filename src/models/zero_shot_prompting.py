from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('./src')
import features
from features.evidence_selection import evidence_triple_selection, triple2text
import pandas as pd
import torch
import json
from llamaapi import LlamaAPI



api_request_json = {
    "messages": [
        {"role": "user", "content": "What is the weather like in Boston?"},
    ],
    "stream": False,
}


def get_llama_api_response(llamaApi, question):
    api_request_json["messages"] = [{"role": "user", "content": question}]
    response = llamaApi.run(api_request_json)
    return response


def load_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):

    maxmem = {
        i: f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range(4)}
    maxmem['cpu'] = '300GB'
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 load_in_8bit=False,
                                                 torch_dtype=torch.float16,
                                                 device_map='auto',
                                                 max_memory=maxmem)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, tokenizer


def formatting_prompt_func(prompt_template, example, max_length=4096):
    return prompt_template.format(
        question=example['question'], context=example['context'][:max_length-len(example['question'])-len(prompt_template)])


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
        prompt_text = formatting_prompt_func(
            prompt_template=prompt_template, example=example, max_length=max_length)
        prompt_texts.append(prompt_text)
    # Return the list of formatted texts
    return prompt_texts


def get_prediction(model, tokenizer, prompt, length=600):

    inputs = tokenizer(prompt, add_special_tokens=True,
                       max_length=1000, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs, max_new_tokens=length)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response


def zero_shot_prompting(model, tokenizer, examples, prompt_template="", evidence_selection=False, verbalizer=False, max_length=4096):
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
        response = get_prediction(model, tokenizer, formatted_prompt)
        responses.append(response[0])

    return responses


def zero_shot_prompting_api(llamaApi, examples, prompt_template="", evidence_selection=False, verbalizer=False, max_length=4096):
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
        response = get_llama_api_response(llamaApi, formatted_prompt)
        responses.append(
            response.json()['choices'][0]['message'])
    return responses


def test_examples(examples, api=False, token=''):

    with open("./data/raw/prompt_template.txt", "r") as file:
        prompt_template = file.read()

    save_file_v_e = "./results/zero-shot-prompting-with-verbalizer-with-evidence-selection.json"
    save_file_v = "./results/zero-shot-prompting-with-verbalizer.json"
    save_file_e = "./results/zero-shot-prompting-with-evidence-selection.json"
    save_file = "./results/zero-shot-prompting.json"

    if (api):
        llama = LlamaAPI(token)
        responses_e = zero_shot_prompting_api(
            llama, examples, prompt_template, evidence_selection=True)
        responses = zero_shot_prompting_api(
            llama, examples, prompt_template)
        responses_v_e = zero_shot_prompting_api(
            llama, examples, prompt_template, evidence_selection=True, verbalizer=True)
        responses_v = zero_shot_prompting_api(
            llama, examples, prompt_template, verbalizer=True)
    else:
        llama3, tokenizer = load_model()
        responses_e = zero_shot_prompting(
            llama3, tokenizer, examples, prompt_template, evidence_selection=True)
        responses = zero_shot_prompting(
            llama3, tokenizer, examples, prompt_template)
        responses_v_e = zero_shot_prompting(
            llama3, tokenizer, examples, prompt_template, evidence_selection=True, verbalizer=True)
        responses_v = zero_shot_prompting(
            llama3, tokenizer, examples, prompt_template, verbalizer=True)
        
    with open(save_file_e, 'w') as file:
        json.dump(responses_e, file)
    with open(save_file, 'w') as file:
        json.dump(responses, file)
    with open(save_file_v_e, 'w') as file:
        json.dump(responses_v_e, file)
    with open(save_file_v, 'w') as file:
        json.dump(responses_v, file)


if __name__ == '__main__':
    with open('./data/processed/test_processed_data.json', 'r') as file:
        examples = json.load(file)
    examples = [examples[0]]
    test_examples(examples, api=True,
                  token="LL-6y4m0SOZr19zOOwik5xOojWL2Sml3i6OBb47AjpMiMKuEy3BMulYsYm6GM4zuI0V")
