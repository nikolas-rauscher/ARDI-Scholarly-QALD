import json
import pandas as pd
import re
def clean_and_save_dataset(input_path, output_path):
    # Load the JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Convert to a DataFrame for easier analysis
    df = pd.DataFrame(data)
    print(f"Number of entries after filtering: {df.shape[0]}")

    df['answer_length'] = df['answer'].apply(len)
    # Filter out answers that are empty or consist of only one character
    filtered_df = df[~((df['answer_length'] <= 1) & ~df['answer'].str.contains(r'\d', na=False))]
    filtered_df = filtered_df.drop(columns=['answer_length'])


    print(f"Number of entries after filtering: {filtered_df.shape[0]}")
    print(filtered_df.head())


    with open(output_path, 'w') as file:
        json.dump(filtered_df.to_dict(orient='records'), file, indent=4)

    print(f"Cleaned dataset saved to: {output_path}")

# Example usage of the function
input_file_path = 'data/raw/dataset/raw_train_dataset.json'
output_file_path = 'data/processed/cleand_dataset/train_dataset.json'
clean_and_save_dataset(input_file_path, output_file_path)
