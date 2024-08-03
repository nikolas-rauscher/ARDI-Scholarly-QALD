from .create_context import create_dataset
import json
import os


def format_author_uris(author_uris: list) -> str:
    """
    Formats a list of author DBLP URIs into a structured string that appears as a serialized JSON object.
    Each URI is given a key based on its position in the list (e.g., 'author1_dblp_uri', 'author2_dblp_uri', etc.).
    
    Args:
        author_uris (list of str): A list containing the DBLP URIs of authors.

    Returns:
        str: A string representing a list containing a single dictionary, with keys and values formatted as specified.
    """
    author_dict = {}
    for index, uri in enumerate(author_uris, start=1):
        key = f"author{index}_dblp_uri"
        author_dict[key] = f"<{uri}>"    
    result = str([author_dict])
    
    return result


def run_question(question: str, author_dblp_uri: list, question_id: str, config) -> dict:
    """
    Processes a question by creating a JSON file with question details and initiates dataset creation.

    Args:
        question (str): The text of the question.
        author_dblp_uri (List[str]): A list of DBLP URIs corresponding to authors of the question.
        question_id (str): A unique identifier for the question, used for naming the saved file.
        config: Configuration object containing paths and URLs used throughout the dataset creation process.

    Return:
        All retrieved triples and relevant wikidata (dict)
    """
    # Create question dictionary
    question_dict = [{
        "id": question_id,
        "question": question,
        "answer": "-",  # Assuming '-' indicates an unanswered question
        "author_dblp_uri": format_author_uris(author_dblp_uri)
    }]

    # Set the file path for saving the question data
    save_path = os.path.join("data/raw", f"{question_id}.json")
    with open(save_path, 'w') as file:
        json.dump(question_dict, file, indent=4, ensure_ascii=False)
    
    # Update the path to the question in the config object
    config.questions_path = save_path  # Ensure this config attribute is correctly used in create_dataset

    # Call create_dataset with the updated config
    create_dataset(config, question_id)
