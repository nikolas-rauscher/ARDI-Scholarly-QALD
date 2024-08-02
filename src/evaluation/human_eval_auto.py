import os
import json

true_answer_file = "data/processed/processed_data.json"
input_folder = 'results/hm_results/todo'
output_folder = 'results/hm_results/hm_evaluations'

# Load true answers
with open(true_answer_file) as f:
    true_answers = json.load(f)


def search_dict_list_by_id(dict_list, id_idx):
    """Find an item in a list of dictionaries by 'id'."""
    return next((item for item in dict_list if item.get('id') == id_idx), None)


def ask_for_confirmation():
    """Ask for confirmation from the user."""
    return input("Please confirm with 'y' or 'n': ").strip().lower() == 'y'


answer_key = {
    0: "correct answer",
    1: "partially correct answer",
    2: "says insufficient information when it is indeed insufficient",
    3: "could not answer question although information was provided",
    4: "wrong answer",
    5: "token limit"
}

if __name__ == "__main__":
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.json'):
            with open(file_path) as f:
                data = json.load(f)

            for item in data:
                true_item = search_dict_list_by_id(true_answers, item['id'])
                if true_item:
                    question = true_item['question']
                    true_answer = true_item['answer']
                    print(
                        f"Question:\n{question}\nTrue Answer:\n{true_answer}\nProvided Answer:\n{item['answer']}\n")

                    # Display feedback options
                    for key, value in answer_key.items():
                        print(f"{key}: {value}")

                    # Get and store human feedback
                    input_str = input(
                        "Enter the number corresponding to your answer: ").strip()
                    if not item.get('human_feedback') or ask_for_confirmation():
                        item['human_feedback'] = answer_key[int(input_str[0])]
                        print(f"Feedback: {item['human_feedback']}")

            # Save updated data
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Processed {filename}")
