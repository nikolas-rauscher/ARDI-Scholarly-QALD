from llamaapi import LlamaAPI
import configparser
import pandas as pd
import torch
import json
import sys
sys.path.append('./src')
from features.evidence_selection import evidence_triple_selection, triple2text
import features
from transformers import AutoModelForCausalLM, AutoTokenizer



config = configparser.ConfigParser()
config.read('config.ini')


api_request_json = {
    "messages": [
        {"role": "user", "content": "What is the weather like in Boston?"},
    ],
    "functions": [
        {
            "name": "get_current_answer",
            "description": "Get the answer given the provided context",
            "parameters": {
                "type": "object",
                "properties": {
                },
            },
            "required": [],
        }
    ],
    "stream": False,
    "function_call": "get_current_answer",
}


def zero_shot_prompting(example, model=None, tokenizer=None, prompt_template="", evidence_selection=False, verbalizer=False, max_length=4096, api=False, llamaApi=None):
    """
    Generate answer to question using a zero-shot learning approach with the provided model and tokenizer.

    Args:
        example (dict): An example containing 'question' (str) and 'all_tripples' (list of triples).
        model (transformers.PreTrainedModel, optional): The pre-trained model for text generation. Defaults to None.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer corresponding to the pre-trained model. Defaults to None.
        prompt_template (str, optional): Template for formatting the prompt. Defaults to an empty string.
        evidence_selection (bool, optional): Flag to determine if evidence selection should be performed. Defaults to False.
        verbalizer (bool, optional): Flag to determine if verbalization should be applied. Defaults to False.
        max_length (int, optional): Maximum length of the generated response. Defaults to 4096.
        api (bool, optional): Flag to determine if the API should be used for response generation. Defaults to False.
        llamaApi (API, optional): API endpoint for LLaMA model response generation. Required if `api` is True. Defaults to None.

    Returns:
        list of str: The generated responses for the input example.

    TODO: Implement verbalizer functionality.
    """
    triples = evidence_triple_selection(
        example['question'], example['all_tripples']) if evidence_selection else example['all_tripples']
    if (not verbalizer):
        example['context'] = ''.join(
            [triple2text(triple) for triple in triples])
    else:
        example['context'] = ""
    formatted_prompt = formatting_prompt_func(
        prompt_template, example, max_length)
    if (api):
        response = get_llama_api_response(llamaApi, formatted_prompt)
        if (response):
            response = response.json()['choices'][0]['message']
    else:
        response = get_prediction(model, tokenizer, formatted_prompt)
        response = response[0]
    print(response)
    return example['context'], response


def get_llama_api_response(llamaApi, question):
    global api_request_json
    api_request_json["messages"] = [{"role": "user", "content": question}]
    try:
        response = llamaApi.run(api_request_json)
    except:
        return None
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


result_sample_template = {
    "id": "",
    "question": "",
    "answer": "",
    "results": {
        "model": "",
        "plain_triples_results": {"response": "", "context": "", "metrics": {"exact_score": "", "meteor": ""}},
        "verbalizer_results": {},
        "evidence_matching": {},
        "verbalizer_plus_evidence_matching": {}
    }
}


def test_examples(examples, prompt_template, model=None, tokenizer=None, model_name="", saved_file_name="", api=False, token='', llamaApi=None):

    responses = []
    global result_sample_template
    result_sample_template["results"]["model"] = model_name
    for example in examples:
        result = result_sample_template.copy()
        result.update({
            "question": example.get("question"),
            "answer": example.get("answer"),
            "id": example.get("id")
        })

        context_plain, response_plain = zero_shot_prompting(
            example, model=model, tokenizer=tokenizer, api=api, prompt_template=prompt_template, llamaApi=llamaApi)
        context_evidence, response_evidence = zero_shot_prompting(example, model=model, tokenizer=tokenizer,
                                                                  api=api, prompt_template=prompt_template, evidence_selection=True, llamaApi=llamaApi)
        context_verbalizer, response_verbalizer = zero_shot_prompting(
            example, model=model, tokenizer=tokenizer, api=api, prompt_template=prompt_template, verbalizer=True, llamaApi=llamaApi)
        context_evidence_verbalizer, response_evidence_verbalizer = zero_shot_prompting(
            example, model=model, tokenizer=tokenizer, api=api, prompt_template=prompt_template, evidence_selection=True, verbalizer=True, llamaApi=llamaApi)

        approach_names = ["plain_triples_results", "evidence_matching",
                          "verbalizer_results", "verbalizer_plus_evidence_matching"]
        responses_gen = [response_plain, response_evidence,
                         response_verbalizer, response_evidence_verbalizer]
        contexts = [context_plain, context_evidence,
                    context_verbalizer, context_evidence_verbalizer]

        for approach_name, res, context in zip(approach_names, responses_gen, contexts):
            result['results'][approach_name] = {
                "context": context,
                "response": res,
                "metrics": {
                    "exact_score": "",
                    "meteor": ""
                }
            }
        responses.append(result)

    with open(saved_file_name, 'w') as file:
        json.dump(responses, file)


if __name__ == '__main__':
    llama3, tokenizer = None, None
    with open(config['FilePaths']['test_data_file'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r')as file:
        prompt_template = file.read()
    examples = [examples[0]]
    llama = LlamaAPI(config['token']['llamaapi'])
    test_examples(examples, prompt_template, api=True, llamaApi=llama, saved_file_name=config["FilePaths"]["zero_shot_prompting_result_file"],
                   model_name="llama3")
