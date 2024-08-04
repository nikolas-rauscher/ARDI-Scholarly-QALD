import numpy as np
from .herlper_functions import read_json
from typing import List, Dict

class DBLP_ANALYSIS():
    
    def __init__(self, DBLP_path_outputdata_preprocessed):
        data = read_json(DBLP_path_outputdata_preprocessed)
        self.data = data

    #get all predicates labels
    def get_all_predicate_labels(data: List[Dict]) -> List[str]:
        """
        Extracts all unique predicate labels from a list triples.

        Parameters:
        - data (List[Dict]): A list of dictionaries, each containing details of questions and their associated triples.

        Returns:
        - List[str]: A list of unique predicate labels from all the triples across all provided triples.

        """
        predicates_labels = []
        predicates = []
        for question, i in zip(data, range(len(data))):
            print(f"Processing record {i}")
            for all_triples in question["all_triples"]:
                for triple in all_triples["triples"]:
                    if triple["predicate"] not in predicates:
                        predicates.append(triple["predicate"])
                        predicates_labels.append(triple["predicateLabel"])
        
        print("Unique predicate labels:", predicates_labels)
        print("Total unique predicates:", len(predicates_labels))
        return predicates_labels


    def get_all_predicates_without_labels(data: List[Dict]) -> List[str]:
        """
        Extracts all unique predicates that do not have labels from a list of data blocks where each block contains triples information.

        Parameters:
        - data (List[Dict]): A list of dictionaries, each containing details of questions and their associated triples.

        Returns:
        - List[str]: A list of unique predicates that do not have associated labels.

        """
        unique_predicates = []
        for question_index, question in enumerate(data):
            print(f"Processing record {question_index}")
            for item in question["all_triples"]:
                if item["predicateLabel"] == "" and item["predicate"] not in unique_predicates:
                    unique_predicates.append(item["predicate"])
        
        print("Predicates without labels:", unique_predicates)
        print("Total predicates without labels:", len(unique_predicates))
        return unique_predicates



    def get_authors_without_orcidID(data: List[Dict]) -> List[str]:
        """
        Parameters:
        - data (List[Dict]): A dataset where each item contains information about triples associated with authors.

        Returns:
        - List[str]: A list of authors (by subject URI) who lack an ORCID ID.
        """
        # Collect all authors identified as 'Person'
        author_list = [item["subject"] for question in data for item in question["all_triples"] if item["objectLabel"] == "Person"]
        authors_with_orcid = set()
        for question in data:
            for item in question["all_triples"]:
                if item["predicateLabel"] == "orcid":
                    authors_with_orcid.add(item["subject"])

        # Filter out authors with an ORCID ID
        filtered_authors = [author for author in author_list if author not in authors_with_orcid]
        print("Count of authors without an ORCID ID:", len(filtered_authors))

        return filtered_authors



def get_average_triples_number_per_question(data: List[Dict]) -> float:
    """
    Parameters:
    - data (List[Dict]): A dataset where each item contains triples associated with a question.

    Returns:
    - float: The average number of triples per question in the dataset.
    """
    triples_counter = []

    # Accumulate the number of triples for each question
    for question in data:
        triples_per_question = 0  # Initialize the counter for triples per question
        for item in question["all_triples"]:
            triples_per_question += len(item["triples"])  # Sum up the number of triples for the current question
        triples_counter.append(triples_per_question)  # Append the total for this question to the list

    average_triples = np.mean(triples_counter)  # Calculate the average number of triples per question
    print("Average number of triples per question:", average_triples)
    
    return average_triples



#get all predicates amount
def get_all_predicates_percentage(data: List[Dict]):
    """
    Calculates the percentage of each unique predicate's occurrence.

    Parameters:
    - data (list): A list of dictionaries, where each dictionary represents data about one question 
      and contains a list of entities, each with its own list of triples.

    Returns:
    - dict: A dictionary where keys are predicates and values are their percentage occurrence across all provided data.
    """
    predicates = {}
    predicates_number = 0
    for question,i in zip(data, range(len(data))):
        for entity in question["all_tripples"]:
            for item in entity["tripples"]: 
                predicates_number += 1
                if item["predicate"] not in predicates:
                    predicates[item["predicate"]]=1
                else:
                    predicates[item["predicate"]]+=1 
    
    for predicate, count in predicates.items():
        predicates[predicate] = (count / predicates_number) * 100
    print(predicates)
    return predicates

        

def main():
    """
    To run this script direcly run:
        python -m src.data_extraction.triple_extraction.dblp.analyse_dataset_dblp    
    from the root directory of this project 
    """

    path = "data/processed/dblp/pre_processed_data10.json"
    dataset = DBLP_ANALYSIS(path)
    dataset.get_all_predicate_labels    
    dataset.get_all_predicate_labels()
    dataset.get_all_predicates_without_labels()
    dataset.get_authors_without_orcidID

##############################################################################
if __name__ == "__main__":
    main()

    


