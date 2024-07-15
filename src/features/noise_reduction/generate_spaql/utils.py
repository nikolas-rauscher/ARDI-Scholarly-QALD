import json
import re


def read_questions(file_name):
    """
    Reads questions from a JSON file and returns them as a list.

    Args:
        file_name (str): The name of the file containing the questions in JSON format.

    Returns:
        list: A list of questions read from the file.
    """
    f = open(file_name)
    questions = json.load(f)
    f.close()
    return questions


def post_process_query(sparql_query):
    """
    Post-processes a SPARQL query to clean and format it for better readability and execution.

    This function removes special characters, extra spaces, and adds a space before question marks in the SPARQL query.

    Args:
        sparql_query (str): The SPARQL query to be post-processed.

    Returns:
        str: The post-processed SPARQL query.
    """
    # Replace '\n' with a space
    cleaned_sparql_query = sparql_query.replace("\n", ' ')
    cleaned_sparql_query = cleaned_sparql_query.replace("\'", '')
    # Replace '\\' with a single '\'
    cleaned_sparql_query = cleaned_sparql_query.replace("\\", '')
    # Replace '\' with an empty string
    cleaned_sparql_query = cleaned_sparql_query.replace("\\", '')
    # Remove extra spaces using regular expressions
    cleaned_sparql_query = re.sub(r'\s+', ' ', cleaned_sparql_query)
    cleaned_string = re.sub(r'\\.', '', cleaned_sparql_query)
    # Use regular expressions to add a space before '?'
    modified_sparql_query = re.sub(r'(\S)\?', r'\1 ?', cleaned_string)
    return modified_sparql_query
