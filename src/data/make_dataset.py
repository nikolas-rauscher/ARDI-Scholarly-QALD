"""Script to download or generate data"""

import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON


def read_json_(trainindata_path) -> dict:
    with open(trainindata_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
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

def create_dataset(trainingdata_path:str, endpoint_url:str, save_processed_data_path: str, outputdata_name: str):
    data = read_json_(trainingdata_path)
    processed_data = []

    for data_block,i in zip(data[:2],range(len(data))):
        print(i)
        list_of_tripples = []
        entities = data_block["author_dblp_uri"]
        if entities[0] == "[":
            entities = eval(entities)[0]
            for key, entity in entities.items():
                #print(retrieve_triples_for_entity(entity,endpoint_url))
                tripples = retrieve_triples_for_entity(entity,endpoint_url)
                list_of_tripples+= tripples
        else:
            #print(retrieve_triples_for_entity(entities,endpoint_url))
            list_of_tripples = retrieve_triples_for_entity(entities,endpoint_url)

        data_block["tripples_number"] = len(list_of_tripples)
        data_block["all_tripples"] = list_of_tripples
        processed_data.append(data_block)

    # Save the new processed training data
    file_path = os.path.join(save_processed_data_path, outputdata_name)
    with open(file_path, 'w') as file:
        json.dump(processed_data, file, indent=4,ensure_ascii=False)



##############################################################################
def main():

    outputdata_name = "processed_data"
    trainingdata_path = "data/raw/trainingdata.json"
    save_processed_data_path = "data/processed"
    endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL
    create_dataset(trainingdata_path, endpoint_url, save_processed_data_path, outputdata_name) 

##############################################################################
if __name__ == "__main__":
    main()

    
            







