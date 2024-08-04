import json
import configparser
import os
from tqdm import tqdm
from groq import Groq
from llamaapi import LlamaAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gc  # Garbage Collector Interface

config = configparser.ConfigParser()
config.read('config.ini')
max_length_input = config.getint('Parameters', 'max_length_input')
max_output_length = config.getint('Parameters', 'max_output_length')

# Approximate characters per token
chars_per_token = 4
max_length_chars = max_length_input * chars_per_token
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def load_model(model_id):
    """
    Load the model and tokenizer for the specified model ID.

    Args:
        model_id (str): The ID of the model to load.

    Returns:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
    """
    if torch.cuda.is_available():
        maxmem = {
            i: f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range(4)}
    else:
        maxmem = {'cpu': '300GB'}
    print(model_id)
    if (("T5" in model_id) or "t5" in model_id):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                      load_in_8bit=False,
                                                      torch_dtype=torch.float16,
                                                      device_map='auto',
                                                      max_memory=maxmem)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     load_in_8bit=False,
                                                     torch_dtype=torch.float16,
                                                     device_map='auto',
                                                     max_memory=maxmem)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return model, tokenizer


def clean_context(context):
    """
    Clean the context by removing special characters.

    Args:
        context (str): The context to be cleaned.

    Returns:
        str: The cleaned context.
    """
    return context.replace('<unk>', ' ')


def truncate_context_to_max_chars(context, question, prompt_template):
    """
    Truncate the context to ensure it fits within the maximum character length.

    Args:
        context (str): The context to truncate.
        question (str): The question for the prompt.
        prompt_template (str): The prompt template.

    Returns:
        str: The truncated context.
    """
    formatted_prompt = prompt_template.format(question=question, context='')
    prompt_length = len(formatted_prompt)
    max_context_length = max_length_chars - prompt_length

    if len(context) > max_context_length:
        print(
            f"Original context is too long and will be truncated. Original length: {len(context)}, truncated length: {max_context_length}")
        context = context[:max_context_length]

    return context


def zero_shot_prompting_local(model, tokenizer, question, context, prompt_template):
    """
    Generate zero-shot predictions using a local model.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
        question (str): The question
        context (str): The context to use from the example.
        prompt_template (str): The prompt template to format the question and context.

    Returns:
        dict: A dictionary containing the role and content of the response.
    """
    context = clean_context(context)
    truncated_context = truncate_context_to_max_chars(
        context, question, prompt_template)
    formatted_prompt = prompt_template.format(
        question=question, context=truncated_context)

    inputs = tokenizer(formatted_prompt, add_special_tokens=True,
                       max_length=max_length_input, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_new_tokens=max_output_length)
    response_text = tokenizer.batch_decode(
        outputs, skip_special_tokens=True)[0]
    return response_text


def zero_shot_prompting_local4settings(model, tokenizer, example, context_type, prompt_template):
    """
    Generate zero-shot predictions using a local model.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
        example (dict): The example containing the question and context.
        context_type (str): The type of context to use from the example.
        prompt_template (str): The prompt template to format the question and context.

    Returns:
        dict: A dictionary containing the role and content of the response.
    """
    return zero_shot_prompting_local(model, tokenizer, question=example["question"], context=example["contexts"][context_type], prompt_template=prompt_template)


def zero_shot_prompting_llama(api, question, context, prompt_template):
    """
    Generate zero-shot predictions using the Llama API.

    Args:
        api (LlamaAPI): The LlamaAPI instance.
        question (str): The question
        context (str): The context to use from the example.
        prompt_template (str): The prompt template to format the question and context.

    Returns:
        dict: A dictionary containing the role and content of the response.
    """
    context = clean_context(context)
    truncated_context = truncate_context_to_max_chars(
        context, question, prompt_template)
    formatted_prompt = prompt_template.format(
        question=question, context=truncated_context)
    response = get_llama_api_response(api, formatted_prompt)
    return response


def zero_shot_prompting_groq(api, question, context, prompt_template, model_id):
    """
    Generate zero-shot predictions using the Groq API.

    Args:
        api (Groq): The Groq API instance.
        question (str): The question
        context (str): The context to use from the example.
        prompt_template (str): The prompt template to format the question and context.
        model_id (str): The ID of the model to use with the Groq API.

    Returns:
        dict: A dictionary containing the role and content of the response.
    """
    context = clean_context(context)
    truncated_context = truncate_context_to_max_chars(
        context, question, prompt_template)
    formatted_prompt = prompt_template.format(
        question=question, context=truncated_context)
    response_text = get_groq_api_response(formatted_prompt, api, model_id)
    return response_text


def get_llama_api_response(api, question):
    """
    Send a request to the Llama API and return the response.

    Args:
        api (LlamaAPI): The LlamaAPI instance.
        question (str): The question to send to the API.

    Returns:
        str: The response from the Llama API.
    """
    api_request_json = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_output_length  # Use max_output_length from config
    }
    try:
        response = api.run(api_request_json)
        if response.status_code != 200:
            return "API call failed."
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        return "API call failed."


def get_groq_api_response(question, client, model_id):
    """
    Send a request to the Groq API and return the response.

    Args:
        question (str): The question to send to the API.
        client (Groq): The Groq API instance.
        model_id (str): The ID of the model to use with the Groq API.

    Returns:
        str: The response from the Groq API.
    """
    try:
        # print(question) only for testing the final Prompt
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=model_id,
            max_tokens=max_output_length  # Use max_output_length from config
        )
        response_text = chat_completion.choices[0].message.content if chat_completion.choices else ""
        print(f"API Response: {response_text}")
        return response_text
    except Exception as e:
        print(f"API call failed: {e}")
        return "API call failed."


def run_model(model_id, model_name, template_files, api=False, api_type=None):
    """
    Run the model to generate zero-shot predictions for the given templates.

    Args:
        model_id (str): The ID of the model to load.
        model_name (str): The name of the model.
        template_files (list): A list of template file paths.
        api (bool, optional): Whether to use an API. Defaults to False.
        api_type (str, optional): The type of API to use ('llama' or 'groq'). Defaults to None.
    """
    llama_api = LlamaAPI(config['Token']['llamaapi']
                         ) if api and api_type == 'llama' else None
    groq_client = Groq(
        api_key=config['Token']['groqapi']) if api and api_type == 'groq' else None
    model, tokenizer = load_model(model_id) if not api else (None, None)
    with open(config['FilePaths']['prepared_data_file'], 'r') as file:
        prepared_data = json.load(file)

    results = []
    for template_file in template_files:
        # Remove comments and whitespace from the file list
        template_file = template_file.split('#')[0].strip()
        if not os.path.exists(template_file):
            print(f"Template file {template_file} does not exist.")
            continue

        with open(template_file, 'r') as file:
            prompt_template = file.read()

        for example in tqdm(prepared_data, desc=f"Generating Zero-Shot Predictions for {api_type} {model_name}"):
            result = {
                "id": example["id"],
                "question": example["question"],
                "true_answer": example["answer"],
                "model": model_name,
                "technique": "zero_shot",
            }

            if api:
                if api_type == 'llama':
                    response = zero_shot_prompting_llama(
                        llama_api, example["question"], example["contexts"], prompt_template)
                elif api_type == 'groq':
                    response = zero_shot_prompting_groq(
                        groq_client, example["question"], example["contexts"], prompt_template, model_id)
            else:
                response = zero_shot_prompting_local(
                    model, tokenizer, example["question"], example["contexts"], prompt_template)

            result["answer"] = response
            results.append(result)

    output_file = config['FilePaths']['default_results_file'].replace(
        ".json", f"_{model_name}.json")
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # Clean up the model and tokenizer to free up memory
    if model:
        del model
    if tokenizer:
        del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def run_model4settings(model_id, model_name, template_files, api=False, api_type=None):
    """
    Run the model to generate zero-shot predictions for the given templates.

    Args:
        model_id (str): The ID of the model to load.
        model_name (str): The name of the model.
        template_files (list): A list of template file paths.
        api (bool, optional): Whether to use an API. Defaults to False.
        api_type (str, optional): The type of API to use ('llama' or 'groq'). Defaults to None.
    """
    if (api):
        llama_api = LlamaAPI(config['Token']['llamaapi']
                             ) if api_type == 'llama' else None
        groq_client = Groq(
            api_key=config['Token']['groqapi']) if api_type == 'groq' else None
    else:
        model, tokenizer = load_model(model_id)
    with open(config['FilePaths']['prepared_data4settings_file'], 'r') as file:
        prepared_data = json.load(file)

    results = []
    for example in tqdm(prepared_data, desc=f"Generating Zero-Shot Predictions for {api_type} {model_name}"):
        result = {
            "id": example["id"],
            "question": example["question"],
            "model": model_name,
            "technique": "zero_shot",
            "results": []
        }

        if ("answer" in example):
            result["true_answer"] = example["answer"]

        for template_file in template_files:
            # Remove comments and whitespace from the file list
            template_file = template_file.split('#')[0].strip()
            if not os.path.exists(template_file):
                print(f"Template file {template_file} does not exist.")
                continue

            with open(template_file, 'r') as file:
                prompt_template = file.read()

            template_result = {
                "prompt_template": template_file,
                "all_triples_results": {
                    "response": {},
                    "metrics": {
                        "exact_score": 0,
                        "meteor": 0
                    }
                },
                "verbalizer_results": {
                    "response": {},
                    "metrics": {
                        "exact_score": 0,
                        "meteor": 0
                    }
                },
                "evidence_matching": {
                    "response": {},
                    "metrics": {
                        "exact_score": 0,
                        "meteor": 0
                    }
                },
                "verbalizer_plus_evidence_matching": {
                    "response": {},
                    "metrics": {
                        "exact_score": 0,
                        "meteor": 0
                    }
                }
            }

            for context_type, context_key in zip(
                ["plain", "verbalizer_on_all_triples", "evidence_matching",
                    "verbalizer_plus_evidence_matching"],
                ["all_triples_results", "verbalizer_results",
                    "evidence_matching", "verbalizer_plus_evidence_matching"]
            ):

                if api:
                    if api_type == 'llama':
                        response = zero_shot_prompting_llama(
                            llama_api, example["question"], example["contexts"][context_type], prompt_template)
                    elif api_type == 'groq':
                        response = zero_shot_prompting_groq(
                            groq_client, example["question"], example["contexts"][context_type], prompt_template, model_id)
                else:
                    response = zero_shot_prompting_local4settings(
                        model, tokenizer, example, context_type, prompt_template)

                template_result[context_key]["response"] = response

            result["results"].append(template_result)

        results.append(result)

    output_file = config['FilePaths']['default4settings_results_file'].replace(
        ".json", f"_{model_name}.json")
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # Clean up the model and tokenizer to free up memory
    if model:
        del model
    if tokenizer:
        del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
