from .helper_function import read_json


def get_all_predicates(data):
    """
    Extracts and returns a list of unique predicates from a dataset containing triple data.

    Parameters:
    - data (list): The dataset from which predicates are to be extracted. Each item should
                   have a 'all_tripples' key containing a list of 'tripples'.

    Returns:
    - predicates (list): A list of unique predicates from the dataset.
    """
    predicates = []
    for question in data:
        for all_triples in question["all_tripples"]:
            for triple in all_triples["tripples"]:
                if triple["predicate"] not in predicates:
                    predicates.append(triple["predicate"])    
    print(predicates)      
    print(len(predicates))
    return predicates

def find_broken_questions(data):
    """
    Prints error messages for questions with incorrect or missing data structure.

    Parameters:
    - data (list): The dataset to check, where each entry is a dictionary representing a question.

    This function directly prints error messages for entries that fail checks and returns None.
    """
    for question in data:
        if "author_dblp_uri" not in question:
            print("ERROR - Missing 'author_dblp_uri' field.")
        
        entities = question["author_dblp_uri"]
        if entities == "":
            print("ERROR - Empty 'author_dblp_uri'.")
        
        if entities[0] != "<" and entities[0] != "[":
            print("ERROR - Invalid format in 'author_dblp_uri'.")

        if entities[0] == "[":
            entities = eval(entities)
            if not entities:
                print("ERROR - Empty list in 'author_dblp_uri'.")

def get_all_predicates_percentage(data):
    """
    Calculates and returns the percentage of occurrence for each predicate in a dataset.

    Parameters:
    - data (list): The dataset from which to calculate predicate percentages.

    Returns:
    - predicates (dict): A dictionary where keys are predicates and values are their
                         percentage of occurrence in the dataset.
    """
    predicates = {}
    predicates_number = 0
    for question in data:
        for entity in question["all_tripples"]:
            for item in entity["tripples"]:
                predicates_number += 1
                if item["predicate"] not in predicates:
                    predicates[item["predicate"]] = 1
                else:
                    predicates[item["predicate"]] += 1
    
    for predicate, count in predicates.items():
        predicates[predicate] = (count / predicates_number) * 100
    print(predicates)
    return predicates
    

data = read_json("data/processed/alex/pre_processed_data10.json")
#res = get_all_predicate_labels(data)
#find_broken_questions(data_raw)
get_all_predicates_percentage(data)
