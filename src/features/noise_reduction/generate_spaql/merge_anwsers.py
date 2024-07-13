import json
import os

def merge_answers(train_dataset_path, sparql_answers_path, output_file_path):
    """
    Merges SPARQL answers into the training dataset by matching the IDs. 
    The answers in the training dataset are replaced with the SPARQL answers.

    Parameters:
    train_dataset_path (str): Path to the training dataset JSON file.
    sparql_answers_path (str): Path to the SPARQL answers JSON file.
    output_file_path (str): Path where the merged dataset will be saved.

    Returns:
    None
    """

    with open(train_dataset_path, 'r') as train_file:
        train_data = json.load(train_file)
        print(f"Loaded {len(train_data)} items from the train dataset")
    

    with open(sparql_answers_path, 'r') as sparql_file:
        sparql_data = json.load(sparql_file)
        print(f"Loaded {len(sparql_data)} items from the SPARQL answers dataset")
    

    sparql_dict = {item['id']: ', '.join(item['sparql_answer']) for item in sparql_data if item['sparql_answer']}
    print(f"Created dictionary with {len(sparql_dict)} SPARQL answers")


    for item in train_data:
        if item['id'] in sparql_dict:
            item['answer'] = sparql_dict[item['id']]
    

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, 'w') as output_file:
        json.dump(train_data, output_file, indent=2)
    
    print(f'Merged data saved to {output_file_path} successfully!')


train_dataset_path = 'src/features/noise_reduction/generate_spaql/datasets/train_dataset.json'
sparql_answers_path = 'src/features/noise_reduction/generate_spaql/datasets/answers/filterd_awnsers/1000_qestions/no_numbers_filterd_awnsers.json'
output_file_path = 'src/features/noise_reduction/generate_spaql/datasets/answers/merged_dataset/1000_qestions/train_dataset_merged_dataset.json'

merge_answers(train_dataset_path, sparql_answers_path, output_file_path)