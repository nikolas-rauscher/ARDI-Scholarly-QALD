import json
import os

# Load the true answers
true_answer_file = 'src/evaluation/nikolas/answers.json'
with open(true_answer_file, encoding='utf-8') as f:
    true_answers = json.load(f)

# Load the zero_shot_prompting_Mistral-7B-Instruct-v0.2.json file
responses_file = 'src/evaluation/nikolas/zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
with open(responses_file, encoding='utf-8') as f:
    responses = json.load(f)

# Function to clear the screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Function to get human feedback
def get_human_feedback(context, question, true_answer, model_answer):
    clear_screen()
    print(f"\033[1;35mContext:\033[0m {context}")
    print(f"\033[1;32mQuestion:\033[0m {question}")
    print(f"\033[1;34mTrue Answer:\033[0m {true_answer}")
    print(f"\033[1;33mModel Answer:\033[0m {model_answer}")
    print("\nPlease provide feedback:")
    print("0: Correct answer")
    print("1: Partially correct answer")
    print("2: Says insufficient information when it is indeed insufficient")
    print("3: Could not answer question although information was provided")
    print("4: Wrong answer")
    
    feedback = input("Enter the feedback number: ")
    return int(feedback)

# Function to split the response into context and answer
def split_response(full_response):
    context_marker = "\nContext:"
    answer_marker = "[/INST] Answer:"
    context = ""
    answer = ""
    
    if context_marker in full_response and answer_marker in full_response:
        context_part = full_response.split(context_marker, 1)[1]
        context = context_part.split(answer_marker, 1)[0].strip()
        answer = context_part.split(answer_marker, 1)[1].strip()
    
    return context, answer

# Iterate through the responses
for response in responses:
    question = response['question']
    true_answer_id = response['id']
    true_answer = true_answers[true_answer_id]
    model = response['model']
    technique = response['technique']
    for result in response['results']:
        prompt_template = result['prompt_template']  # Assuming this is the context path, adjust if necessary
        for category in ['all_triples_results', 'verbalizer_results', 'evidence_matching', 'verbalizer_plus_evidence_matching']:
            full_response = result[category]['response']
            context, model_answer = split_response(full_response)
            
            # Ensure that the question, true_answer, context, and model_answer are not empty before prompting feedback
            if not question or not true_answer or not context or not model_answer:
                print(f"Debug: Missing data for category {category}. Skipping feedback prompt.")
                print(f"Question: {question}")
                print(f"True Answer: {true_answer}")
                print(f"Context: {context}")
                print(f"Model Answer: {model_answer}")
                continue
            
            feedback = get_human_feedback(context, question, true_answer, model_answer)
            
            # Update the metrics with human feedback
            result[category]['metrics']['human_feedback'] = feedback
            result[category]['context'] = context
            result[category]['answer'] = model_answer
            # Delete the old response field
            del result[category]['response']

# Save the updated responses with human feedback
updated_responses_file = 'human_eval_updated_zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
with open(updated_responses_file, 'w', encoding='utf-8') as f:
    json.dump(responses, f, ensure_ascii=False, indent=4)

print(f"Updated responses saved to {updated_responses_file}")