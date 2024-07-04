"""Script to download or generate data"""

import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON


def read_json_(trainindata_path, subset) -> dict:
    with open(trainindata_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    if subset: 
        formulations = formulations[:subset]
    return formulations

from SPARQLWrapper import SPARQLWrapper, JSON

def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> list:
    sparql = SPARQLWrapper(endpoint_url)

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

     # Set the query and the return format
    sparql.setQuery(query_subject)
    sparql.setReturnFormat(JSON)
    # Execute the query and convert the results to a Python dictionary
    results = sparql.query().convert()
    list_of_triples = []
    for result in results["results"]["bindings"]:
        triple = {}
        triple["subject"] = entity
        triple["predicate"] = result["predicate"]["value"]
        triple["object"] = result["object"]["value"]
        triple["subjectLabel"] = result.get("subjectLabel", {}).get("value", "")
        triple["predicateLabel"] = result.get("predicateLabel", {}).get("value", "")
        triple["objectLabel"] = result.get("objectLabel", {}).get("value", "")
        list_of_triples.append(triple)


    # Set the query and the return format
    sparql.setQuery(query_object)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        triple = {}
        triple["subject"] = result["subject"]["value"]
        triple["predicate"] = result["predicate"]["value"]
        triple["object"] = entity
        triple["subjectLabel"] = result.get("subjectLabel", {}).get("value", "")
        triple["predicateLabel"] = result.get("predicateLabel", {}).get("value", "")
        triple["objectLabel"] = result.get("objectLabel", {}).get("value", "")
        list_of_triples.append(triple)
    
 
    return list_of_triples

def create_dataset_dblp(trainingdata_path:str, endpoint_url:str, save_processed_data_path: str, outputdata_name: str, subset = None):
    data = read_json_(trainingdata_path, subset)
    processed_data = []

    for data_block,i in zip(data,range(len(data))):
        print(i)
        list_of_tripples_for_authors= []
        tripples_for_one_author = []
        entities = data_block["author_dblp_uri"]
        if entities[0] == "[":
            entities = eval(entities)[0]
            for key, entity in entities.items():
                dic_for_one_author = {}
                tripples_for_one_author = retrieve_triples_for_entity(entity,endpoint_url)
                dic_for_one_author["entity"] = entity
                dic_for_one_author["tripples"] = tripples_for_one_author
                list_of_tripples_for_authors.append(dic_for_one_author)
        else:
            tripples_for_one_author = retrieve_triples_for_entity(entities,endpoint_url)
            dic_for_one_author = {}
            dic_for_one_author["entity"] = entities
            dic_for_one_author["tripples"] = tripples_for_one_author
            list_of_tripples_for_authors.append(dic_for_one_author)
        all_tripples_length  = 0
        for author in list_of_tripples_for_authors:
            all_tripples_length += len(author["tripples"])
        data_block["tripples_number"] = all_tripples_length
        data_block["all_tripples"] = list_of_tripples_for_authors
        processed_data.append(data_block)

    # Save the new processed training data
    file_path = os.path.join(save_processed_data_path, outputdata_name)
    with open(file_path, 'w') as file:
        json.dump(processed_data, file, indent=4,ensure_ascii=False)



##############################################################################
def main():

    outputdata_name = "pre_processed_data1000.json"
    trainingdata_path = "data/raw/trainingdata.json"
    save_processed_data_path = "data/processed/dblp"
    endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL
    create_dataset_dblp(trainingdata_path, endpoint_url, save_processed_data_path, outputdata_name, subset = 1000) 

##############################################################################
if __name__ == "__main__":
    main()

    
            







""""

[
    {
        "id": "6b8aa79c-3908-4f03-b85b-aa1a325d9fe6",
        "question": "What type of information sources were found to be lacking in organized information at Social Services offices according to the author's observation?",
        "answer": "oral communication and notes",
        "tripples_number": 885,
        "all_tripples": [
        {
                {
                    "entity":"<https://dblp.org/pid/w/TDWilson>", 
                    "all_tripples": [
                        {
                            "subject": "<https://dblp.org/pid/w/TDWilson>",
                            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                            "object": "https://dblp.org/rdf/schema#Creator",
                            "subjectLabel": "Thomas D. Wilson 0001",
                            "predicateLabel": "",
                            "objectLabel": "Creator"
                        }, 
                    ]}, 
                {
                    "entity":"<https://dblp.org/pid/w/TDWilson>", 
                    "all_tripples": [ 
                        {
                            "subject": "<https://dblp.org/pid/w/TDWilson>",
                            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                            "object": "https://dblp.org/rdf/schema#Creator",
                            "subjectLabel": "Thomas D. Wilson 0001",
                            "predicateLabel": "",
                            "objectLabel": "Creator"
                        }, 
                    ]
                }
            ]
        }
    }
]




"""