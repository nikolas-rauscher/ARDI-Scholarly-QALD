import json
import os
from .herlper_functions import read_json

take_label= ["authored by", "edited by", "differentFrom"]
take_object=["affiliation","primary affiliation", "creator note"]
link= ["wikidata","homepage URL","primary homepage URL","web page URL", "wikipedia page URL", "archived page URL", "award page URL"]
predicate= ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
skip_labels= ["signature creator", "sameAs", "full creator name", "orcid", "primary full creator name"]
skip_uris = ["http://www.w3.org/2000/01/rdf-schema#label", "http://purl.org/spar/datacite/hasIdentifier"]


def post_process_dblp(outputdata_name: str, pre_processed_data_path: str):
    """
    Processes DBLP data to reformat and filter triples based on specified conditions,
    and then saves the modified data.

    Parameters:
    - outputdata_name (str): The name of the file to save the processed data to.
    - pre_processed_data_path (str): Path to the pre-processed data file.

    The function reads a JSON file containing questions and their associated triples,
    filters and transforms these triples based on specific rules, and saves the new structured
    data into another JSON file in a specified directory.
    """

    data = read_json(pre_processed_data_path)

    post_processed_data= []

    for question in data:
        questions_dict={}
        questions_dict["id"] = question["id"]
        questions_dict["question"] = question["question"]
        questions_dict["answer"] = question["answer"]
        questions_dict["author_dblp_uri"] = question["author_dblp_uri"]
        questions_dict["triples_number"] = 0 
        questions_dict["all_triples"] = []

        for entity in question["all_triples"]:
            dic_for_one_author = {}
            
            new_triples= []
            for tripple in entity["triples"]:
                new_tripple_dict = {}
                if tripple["predicate"] in predicate:
                    new_tripple_dict["subject"] = tripple["subjectLabel"]
                    new_tripple_dict["predicate"] = "is"
                    new_tripple_dict["object"] = tripple["objectLabel"]
                elif tripple["predicateLabel"] in take_label:
                    new_tripple_dict["subject"] = tripple["subjectLabel"]
                    new_tripple_dict["predicate"] = tripple["predicateLabel"]
                    new_tripple_dict["object"] = tripple["objectLabel"]
                elif tripple["predicateLabel"] in take_object:
                    new_tripple_dict["subject"] = tripple["subjectLabel"]
                    new_tripple_dict["predicate"] = tripple["predicateLabel"]
                    new_tripple_dict["object"] = tripple["object"]
                elif tripple["predicateLabel"] in link:
                    new_tripple_dict["subject"] = tripple["subjectLabel"]
                    new_tripple_dict["predicate"] = "link"
                    new_tripple_dict["object"] = tripple["object"]
                elif tripple["predicateLabel"] in skip_labels or tripple["predicate"] in skip_uris :
                    pass 
                else:
                    print(f"undefined id: {tripple['predicate']} for questionid: {question['id']}")
                if new_tripple_dict: new_triples.append(new_tripple_dict)
            
            dic_for_one_author["entity"] = entity["entity"]
            dic_for_one_author["triples"] = new_triples


            questions_dict["all_triples"].append(dic_for_one_author)
        
        all_triples_length  = 0
        for author in questions_dict["all_triples"]:
            all_triples_length += len(author["triples"])
        questions_dict["triples_number"] = all_triples_length

        post_processed_data.append(questions_dict)

    # Save the new processed training data
    save_processed_data_path = "data/processed/dblp"
    file_path = os.path.join(save_processed_data_path, outputdata_name)
    with open(file_path, 'w') as file:
        json.dump(post_processed_data, file, indent=4,ensure_ascii=False)


def main():
    """
    To run this script direcly run:
        python -m src.data_extraction.triple_extraction.dblp.postprocessing_dblp    
    from the root directory of this project 
    """

    pre_processed_data_path = "data/processed/dblp/pre_processed_data10.json"
    outputdata_name = "post_processed_data10.json"

    post_process_dblp(outputdata_name, pre_processed_data_path) 

##############################################################################
if __name__ == "__main__":
    main()


                 

             
            