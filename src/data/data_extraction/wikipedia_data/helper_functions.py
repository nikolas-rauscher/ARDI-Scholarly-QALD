import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def ngram_cosine_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Calculates the cosine similarity between two strings based on their character n-grams.

    Parameters:
    - s1 (str): The first string for comparison.
    - s2 (str): The second string for comparison.
    - n (int, optional): The size of n-gram to be used. Default is 3.

    Returns:
    - float: The cosine similarity score between the two strings, ranging from 0 to 1,
      where 0 means no similarity and 1 means complete similarity.
    """
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([s1, s2])
    return cosine_similarity(ngrams)[0, 1]


def read_json(outputdata_name_path: str) -> dict:
    """
    Parameters:
    - trainindata_path (str): The file path to the JSON file to be read.

    Returns:
    - formulations (List[Dict]): A list of dictionaries parsed from the JSON file. 
    """
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations


def save_intermediate_result(outputdata_path: str, new_dataset: dict) -> None:
    """
    Args:
        outputdata_path (str): The file path where the dataset should be saved.
        new_dataset (dict): The dataset to be saved
    """
    with open(outputdata_path, 'w', encoding='utf-8') as file:
        json.dump(new_dataset, file, indent=4, ensure_ascii=False)
        