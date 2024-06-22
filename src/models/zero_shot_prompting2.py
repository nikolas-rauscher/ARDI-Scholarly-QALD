import json
import configparser
import torch
import sys
sys.path.append('./src')
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamaapi import LlamaAPI

config = configparser.ConfigParser()
config.read('config.ini')

def load_model(model_id):
    """
    Load the model and tokenizer for the specified model ID.
    
    Args:
        model_id (str): The ID of the model to load.
        
    Returns:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
    """
    maxmem = {i: f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range(4)}
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

def zero_shot_prompting(model, tokenizer, example, context_type, prompt_template, max_length=4096, api=False, llamaApi=None):
    """
    Generate zero-shot predictions based on the provided context and prompt template.
    
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
        example (dict): The example containing the question and context.
        context_type (str): The type of context to use from the example.
        prompt_template (str): The prompt template to format the question and context.
        max_length (int, optional): The maximum length of the context. Defaults to 4096.
        api (bool, optional): Whether to use the Llama API. Defaults to False.
        llamaApi (LlamaAPI, optional): The LlamaAPI instance if using the API. Defaults to None.
        
    Returns:
        tuple: A tuple containing the context and the generated response.
    """
    context = example["contexts"][context_type]
    formatted_prompt = prompt_template.format(question=example['question'], context=context[:max_length-len(example['question'])-len(prompt_template)])
    print(f"Formatted Prompt: {formatted_prompt}")
    
    if api:
        response = get_llama_api_response(llamaApi, formatted_prompt)
        if response:
            try:
                response_json = response.json()
                print(f"API Response: {response_json}")  # Log the response
                response = response_json['choices'][0]['message']['content']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                response = {"role": "assistant", "content": "API call failed to return valid JSON."}
            except Exception as e:
                print(f"Unexpected error processing API response: {e}")
                response = {"role": "assistant", "content": "API call failed with an unexpected error."}
        else:
            response = {"role": "assistant", "content": "API call failed."}
    else:
        inputs = tokenizer(formatted_prompt, add_special_tokens=True, max_length=max_length, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs, max_new_tokens=600)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response = {"role": "assistant", "content": response}
    
    return context, response

def get_llama_api_response(llamaApi, question):
    """
    Send a request to the Llama API and return the response.
    
    Args:
        llamaApi (LlamaAPI): The LlamaAPI instance.
        question (str): The question to send to the API.
        
    Returns:
        requests.Response: The response from the Llama API.
    """
    api_request_json = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 3000  # Increase max tokens to allow longer responses
    }
    try:
        print(f"API Request: {api_request_json}")  # Log the request
        response = llamaApi.run(api_request_json)
        if response.status_code != 200:
            print(f"API call failed with status code: {response.status_code}")
            print(f"API Response: {response.text}")  # Log the response text for debugging
            return None
        print("API request successful")
        return response
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def run_model(model_id, model_name, template_files, api=False, llamaApi=None):
    """
    Run the model to generate zero-shot predictions for the given templates.
    
    Args:
        model_id (str): The ID of the model to load.
        model_name (str): The name of the model.
        template_files (list): A list of template file paths.
        api (bool, optional): Whether to use the Llama API. Defaults to False.
        llamaApi (LlamaAPI, optional): The LlamaAPI instance if using the API. Defaults to None.
    """
    model, tokenizer = load_model(model_id) if not api else (None, None)
    with open(config['FilePaths']['prepared_data_file'], 'r') as file:
        prepared_data = json.load(file)
    
    results = []
    for example in tqdm(prepared_data, desc=f"Generating Zero-Shot Predictions for {model_name}"):
        result = {
            "id": example["id"],
            "question": example["question"],
            "true_answer": example["answer"],
            "model": model_name,
            "technique": "zero_shot",
            "results": []
        }
        
        for template_file in template_files:
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
                ["plain", "verbalizer_on_all_tripples", "evidence_matching", "verbalizer_plus_evidence_matching"],
                ["all_triples_results", "verbalizer_results", "evidence_matching", "verbalizer_plus_evidence_matching"]
            ):
                context, response = zero_shot_prompting(model, tokenizer, example, context_type, prompt_template, api=api, llamaApi=llamaApi)
                template_result[context_key]["response"] = response
            
            result["results"].append(template_result)
        
        results.append(result)
    
    output_file = config['FilePaths']['zero_shot_results_file'].replace(".json", f"_{model_name}.json")
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)  # Indent added for better formatting

if __name__ == '__main__':
    llama = LlamaAPI(config['Token']['llamaapi']) if config['Flags']['use_api'] == 'True' else None
    api = config['Flags']['use_api'] == 'True'
    
    model_templates = {
        config['Model']['llama3_id']: config['Templates']['llama3_templates'].split(', '),
        config['Model']['llama2_id']: config['Templates']['llama2_templates'].split(', '),
        # config['Model']['mistral_id']: config['Templates']['mistral_templates'].split(', ')
    }
    
    for model_id, templates in model_templates.items():
        model_name = model_id.split('/')[-1]
        run_model(model_id, model_name, templates, api=api, llamaApi=llama)