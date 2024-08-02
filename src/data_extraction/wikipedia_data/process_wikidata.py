import ast
import json
import os

def format_man(input_str: str) -> str:
    """
    Cleans and formats a malformed JSON string to make it valid.

    Args:
        input_str (str): The malformed JSON string.

    Returns:
        str: A properly formatted JSON string.
    """
    json_str = input_str.replace("'", "")
    json_str = json_str.replace('"', "")
    json_str = json_str.replace("\\n", "")
    json_str = json_str.replace('\\', '')
    json_str = json_str.replace("institute_wikipedia_text:", '"institute_wikipedia_text":"')
    json_str = json_str.replace("'author_wikipedia_text'", '"author_wikipedia_text":"')
    json_str= json_str+ '"}]'
    return json_str


def prepare_wikipedia_data(path_wikipedia_data: str):
    """
    Processes raw Wikipedia data from a specified path, fixes any malformed JSON, and saves the cleaned data.

    Args:
        path_wikipedia_data (str): The file path where the raw Wikipedia data is stored.

    Raises:
        FileNotFoundError: If the raw data file is not found at the specified path.
    """
    try:    
        with open(path_wikipedia_data, "r") as fout:
            data = fout.readlines()
    except FileNotFoundError:
        print(f"Error: The directory '{os.path.dirname(path_wikipedia_data)}' does not exist. Raw Wikipidiadata file is missing")

    data_dict = []
    for i in range(len(data)):
        try:
            output = ast.literal_eval(data[i])
            output = ast.literal_eval(output) 
            data_dict.append(output)
        except:
            output = format_man(output)
            output = json.loads(output)
            data_dict.append(output)

    save_path = "data/processed/wikipedia_data/wikipedia_data_processed" 
    with open(save_path, 'w') as file:
        json.dump(data_dict, file, indent=4, ensure_ascii=False)
    
