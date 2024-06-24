import os
import json
true_answer_file = "data/processed/processed_data.json"
folder_path = 'results/hm_results/processed'
out_folder_path = 'hm'
with open(true_answer_file) as f:
    true_answers = json.load(f)


def search_dict_list_by_id(dict_list, id_idx):
    found_item = None
    for item in dict_list:
        if item.get('id') == id_idx:
            found_item = item
            break
    return found_item


def ask_for_confirmation():
    return input("Please confirm with 'y' or 'n': ").strip().lower() == 'y'


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path) and filename.endswith('.json'):
        with open(file_path) as f:
            data = json.load(f)
            for item in data:
                true_item = search_dict_list_by_id(true_answers, item['id'])
                question = true_item['question']
                true_answer = true_item['answer']
                print(
                    f"question:\n{question}\ntrue_answer:\n{true_answer}\n{filename} answer:\n{item['answer']}\n ")
                input_str = input("human feedback:\n")
                if ((not 'human_feedback' in item) or ((ask_for_confirmation()))):
                    item['human_feedback'] = input_str

        with open(os.path.join(out_folder_path, filename)) as f:
            json.dump(data, f, indent=2)
