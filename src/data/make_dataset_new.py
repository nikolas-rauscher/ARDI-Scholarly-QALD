"""Script to download or generate data"""

"""Script to download or generate data"""

import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON


trainingdata_path = "data/raw/trainingdata.json"
save_processed_data_path = "data/processed"
endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL

def read_json_(trainindata_path) -> dict:
    with open(trainindata_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

from SPARQLWrapper import SPARQLWrapper, JSON

def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> list:
    sparql = SPARQLWrapper(endpoint_url)

    # Extending the query to fetch labels for entities
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?subject ?predicate ?object ?subjectLabel ?predicateLabel ?objectLabel
        WHERE {{
            {{ {entity} ?predicate ?object }}
            UNION
            {{ ?subject ?predicate {entity} }}
            OPTIONAL {{ {entity} rdfs:label ?subjectLabel }}
            OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
            OPTIONAL {{ ?object rdfs:label ?objectLabel }}
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)  # Corrected method name here
    results = sparql.query().convert()

    list_of_triples = []
    for result in results["results"]["bindings"]:
        triple = {
            "subject": result.get("subject", {}).get("value", entity),  # Corrected default to entity if not present
            "predicate": result["predicate"]["value"],
            "object": result.get("object", {}).get("value", entity),  # Corrected default to entity if not present
            "subject_label": result.get("subjectLabel", {}).get("value", ""),
            "predicate_label": result.get("predicateLabel", {}).get("value", ""),
            "object_label": result.get("objectLabel", {}).get("value", "")
        }
        list_of_triples.append(triple)

    return list_of_triples

data = read_json_(trainingdata_path)
processed_data = []

for data_block in data[:10]:
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
file_path = os.path.join(save_processed_data_path, "processed_data.json")
with open(file_path, 'w') as file:
    json.dump(processed_data, file, indent=4,ensure_ascii=False)



    
    
            





