import json

def filter_non_empty_sparql_answers(input_file_path, output_file_path):
    """
    Filters out items with empty 'sparql_answer' from the input JSON file and saves the filtered data to a new JSON file.

    Args:
        input_file_path (str): The path to the input JSON file containing the data to be filtered.
        output_file_path (str): The path where the filtered data will be saved.

    Returns:
        None
    """
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)
    
    # Filter the data to keep only items with non-empty sparql_answer
    filtered_data = [item for item in data if item['sparql_answer']]

    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=2)
    
    print(f'Filtered data saved to {output_file_path} successfully!')

def filter_non_empty_non_numeric_sparql_answers(input_file_path, output_file_path):
    """
    Filters out items with empty or numeric-only 'sparql_answer' from the input JSON file and saves the filtered data to a new JSON file.

    Args:
        input_file_path (str): The path to the input JSON file containing the data to be filtered.
        output_file_path (str): The path where the filtered data will be saved.

    Returns:
        None
    """
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)
        
    def contains_only_numbers(lst):
        """
        Checks if all characters in a string are digits.

        Args:
            lst (str): The string to check.

        Returns:
            bool: True if all characters are digits, False otherwise.
        """
        return all(item.isdigit() for item in lst)
    
    filtered_data = [
        item for item in data 
        if item['sparql_answer'] and not contains_only_numbers(item['sparql_answer'])
    ]
    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=2)
    
    print(f'Filtered data saved to {output_file_path} successfully!')




input_file_path = 'src/features/noise_reduction/generate_spaql/datasets/answers/answer_without_schema_3_shot_1000_Questions_20240712.json'
output_file_path_all_answers = 'src/features/noise_reduction/generate_spaql/datasets/answers/filterd_awnsers/1000_qestions/all_filterd_awnsers.json'
output_file_path_no_numbers = 'src/features/noise_reduction/generate_spaql/datasets/answers/filterd_awnsers/1000_qestions/no_numbers_filterd_awnsers.json'

# Filter all non-empty sparql answers
filter_non_empty_sparql_answers(input_file_path, output_file_path_all_answers)

# Filter non-empty sparql answers without numbers
filter_non_empty_non_numeric_sparql_answers(input_file_path, output_file_path_no_numbers)