import json
import os
from SPARQLWrapper import  JSON
from .herlper_functions import read_json, get_triples_for_entity
from typing import List, Dict
from typing import Optional



def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> List[Dict]:
    """
    Retrieves RDF triples associated with a specified entity from a SPARQL endpoint.

    Parameters:
    - entity (str): The URI of the entity for which to retrieve triples.
    - endpoint_url (str): The URL of the SPARQL endpoint.

    Returns:
    - list_of_triples (List[Dict]): A list of dictionaries where each dictionary represents a triple involving the specified entity. The list includes triples where the entity is either a subject or an object.

    This function constructs two SPARQL queries to retrieve triples:
    1. Where the entity is a subject, fetching associated objects and labels.
    2. Where the entity is an object, fetching associated subjects and labels.
    The function returns a combined list of triples from both queries.
    """

    # Extending the query to fetch labels for entities
    query_subject = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX dblp: <https://dblp.org/rdf/schema-2017-04-18#>

    SELECT ?predicate ?object ?predicateLabel ?subjectLabel ?objectLabel
    WHERE {{
        {entity} ?predicate ?object.
        OPTIONAL {{ {entity} rdfs:label ?subjectLabel }}
        OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
        OPTIONAL {{ ?object rdfs:label ?objectLabel }}
    }}
"""

    query_object = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX dblp: <https://dblp.org/rdf/schema-2017-04-18#>

        SELECT ?predicate ?subject ?predicateLabel ?subjectLabel ?objectLabel
        WHERE {{
            ?subject ?predicate {entity}.
            OPTIONAL {{ ?subject rdfs:label ?subjectLabel }}
            OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
            OPTIONAL {{ {entity} rdfs:label ?objectLabel }}
        }}
    """

    list_of_triples_subject= get_triples_for_entity(query_subject,endpoint_url, "subject", entity)
    list_of_triples_object = get_triples_for_entity(query_object,endpoint_url, "object", entity)

    list_of_triples = list_of_triples_subject + list_of_triples_object
    return list_of_triples

def create_dataset_dblp(trainingdata_path: str, endpoint_url: str, outputdata_path: str, subset: Optional[int] = None, direct_input = False):
    """
    Processes a dataset by reading JSON data, retrieving triples for each entity mentioned,
    and saving the processed data into a new JSON file.

    Parameters:
    - trainingdata_path (str): Path to the raw training data JSON file.
    - endpoint_url (str): URL of the SPARQL endpoint to fetch triples.
    - save_processed_data_path (str): Directory path where the processed data will be saved.
    - outputdata_name (str): Name for the output file that will contain the processed data.
    - direct_input ((Optional[Bool]): Flag indicating whether data is passed directly or as paths to data file. 
    - subset (Optional[int]): A subset of the data to process, useful for debugging or limited runs.
    
    The function reads a JSON file containing data blocks, each with an "author_dblp_uri" key that might contain multiple entities.
    It fetches RDF triples for each entity using the `retrieve_triples_for_entity` function and stores the results.
    It calculates the total number of triples retrieved for each data block and appends this information along with all retrieved triples.
    Finally, it saves the enriched dataset to a new JSON file in the specified directory.
    """
    print("Extracting triples for DBLP KG...\n")

    if direct_input:
        data = direct_input
    else:
        data = read_json(trainingdata_path, subset)
    processed_data = []

    for data_block,i in zip(data,range(len(data))):
        print(i,"/",len(data))
        list_of_tripples_for_authors= []
        tripples_for_one_author = []
        entities = data_block["author_dblp_uri"]
        if entities[0] == "[":
            entities = eval(entities)[0]
            for _, entity in entities.items():
                dic_for_one_author = {}
                tripples_for_one_author = retrieve_triples_for_entity(entity,endpoint_url)
                dic_for_one_author["entity"] = entity
                dic_for_one_author["triples"] = tripples_for_one_author
                list_of_tripples_for_authors.append(dic_for_one_author)
        else:
            tripples_for_one_author = retrieve_triples_for_entity(entities,endpoint_url)
            dic_for_one_author = {}
            dic_for_one_author["entity"] = entities
            dic_for_one_author["triples"] = tripples_for_one_author
            list_of_tripples_for_authors.append(dic_for_one_author)
        all_tripples_length  = 0
        for author in list_of_tripples_for_authors:
            all_tripples_length += len(author["triples"])
        data_block["triples_number"] = all_tripples_length
        data_block["all_triples"] = list_of_tripples_for_authors
        processed_data.append(data_block)

    # Save the new processed training data
    if not os.path.exists("./data/interim/dblp"):
        os.makedirs("./data/interim/dblp")
    with open(outputdata_path, 'w') as file:
        json.dump(processed_data, file, indent=4,ensure_ascii=False)

    print("Finished extracting triples for DBLP KG\n")


##############################################################################
def main():
    """
    To run this script direcly run:
        python -m src.data.data_extraction.triple_extraction.dblp.create_dataset_dblp
    from the root directory of this project 
    """
    outputdata_path = "data/interim/dblp/pre_processed_data10.json"
    trainingdata_path = "data/raw/questions.json"
    endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL
    create_dataset_dblp(trainingdata_path, endpoint_url, outputdata_path, subset = 10) 

##############################################################################
if __name__ == "__main__":
    main()

