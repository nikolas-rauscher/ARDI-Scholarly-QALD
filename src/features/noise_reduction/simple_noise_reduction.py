import json
import pandas as pd
import re

def clean_and_save_dataset(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)


    df = pd.DataFrame(data)
    print(f"Number of entries before filtering: {df.shape[0]}")

    # Calculate the length of each answer
    df['answer_length'] = df['answer'].apply(len)

    # Filter out answers that are empty, consist of only one character and don't contain numbers, or are "N/A"
    filtered_df = df[~((df['answer_length'] <= 1) & ~df['answer'].str.contains(r'\d', na=False))]
    filtered_df = filtered_df[filtered_df['answer'] != "N/A"]
    filtered_df = filtered_df.drop(columns=['answer_length'])

    print(f"Number of entries after filtering: {filtered_df.shape[0]}")
    print(filtered_df.head())

    # Save the cleaned dataset
    with open(output_path, 'w') as file:
        json.dump(filtered_df.to_dict(orient='records'), file, indent=4)

    print(f"Cleaned dataset saved to: {output_path}")


input_file_path = 'src/features/noise_reduction/generate_spaql/datasets/answers/merged_dataset/1000_qestions/processed_data500_merged_dataset.json'
output_file_path = 'data/processed/cleand_dataset/processed_data500_cleand_and_merged_train_dataset.json'
clean_and_save_dataset(input_file_path, output_file_path)