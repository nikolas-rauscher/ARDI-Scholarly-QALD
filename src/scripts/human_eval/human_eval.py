import json
import os

# Configuration
TRUE_ANSWER_FILE = 'results/evaluation/scripts_for_human_eval/input/answers.json'
RESPONSES_FILE = 'results/zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
UPDATED_RESPONSES_FILE = 'results/evaluation/scripts_for_human_eval/results/human_eval_zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
FEEDBACK_MAPPING = {
    0: 'Correct',
    1: 'Partially Correct',
    2: 'Insufficient Info Correct',
    3: 'Insufficient Info Wrong',
    4: 'Wrong',
    'Token limit': 'Token limit'
}

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_human_feedback(context, question, true_answer, model_answer):
    """
    Display the context, question, true answer, and model answer, then prompt for human feedback.

    Args:
        context (str): The context of the question.
        question (str): The question text.
        true_answer (str): The correct answer.
        model_answer (str): The model-generated answer.

    Returns:
        int: The feedback code provided by the user.
    """
    clear_screen()
    print(f"\033[1;35mContext:\033[0m {context}")
    print(f"\033[1;32mQuestion:\033[0m {question}")
    print(f"\033[1;34mTrue Answer:\033[0m {true_answer}")
    print(f"\033[1;33mModel Answer:\033[0m {model_answer}")
    print("\nPlease provide feedback:")
    for code, description in FEEDBACK_MAPPING.items():
        print(f"{code}: {description}")

    while True:
        try:
            feedback = int(input("Enter the feedback number: ").strip())
            if feedback in range(5):
                return feedback
            print("Please enter a valid number between 0 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 4.")


def split_response(full_response):
    """
    Split the full response into context and answer.

    Args:
        full_response (str): The full response containing context and answer.

    Returns:
        tuple: A tuple containing the context and the answer.
    """
    context_marker = "Context:"
    answer_marker = "Answer:"
    context, answer = "", ""

    if context_marker in full_response and answer_marker in full_response:
        context_part = full_response.split(context_marker, 1)[1]
        context = context_part.split(answer_marker, 1)[0].strip()
        answer = context_part.split(answer_marker, 1)[1].strip()

    return context, answer


def process_responses(true_answers, responses):
    """Process responses and collect human feedback."""
    for response in responses:
        question = response['question']
        true_answer_id = response['id']
        true_answer = true_answers.get(true_answer_id, "")
        for result in response['results']:
            for category in ['all_triples_results', 'verbalizer_results', 'evidence_matching', 'verbalizer_plus_evidence_matching']:
                context, model_answer = split_response(
                    result[category]['response'])
                if not question or not true_answer or not context or not model_answer:
                    print(
                        f"Debug: Missing data for category {category}. Skipping feedback prompt.")
                    continue

                feedback = get_human_feedback(
                    context, question, true_answer, model_answer)
                result[category]['metrics']['human_feedback'] = feedback
                result[category]['response'] = {
                    "Context": context, "Answer": model_answer}


def save_responses(responses, output_file):
    """Save the updated responses with human feedback."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
    print(f"Updated responses saved to {output_file}")


def main():
    true_answers = load_json(TRUE_ANSWER_FILE)
    responses = load_json(RESPONSES_FILE)
    process_responses(true_answers, responses)
    save_responses(responses, UPDATED_RESPONSES_FILE)


def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    main()
