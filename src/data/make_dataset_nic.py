"""Script to download or generate data"""

import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm


trainingdata_path = "data/raw/trainingdata.json"
save_processed_data_path = "data/processed"
endpoint_url = "https://dblp-april24.skynet.coypu.org/sparql"  # SPARQL endpoint URL


def read_json_(trainindata_path) -> dict:
    with open(trainindata_path, "r", encoding="utf-8") as file:
        formulations = json.load(file)
    return formulations


def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> list:

    sparql = SPARQLWrapper(
        endpoint_url
    )  # Initialize the SPARQL wrapper with the endpoint URL

    # SPARQL query
    query = f"""
        PREFIX dblp: <https://dblp.org/rdf/schema-2017-04-18#>

        SELECT ?predicate ?object ?subject
        WHERE {{
            {{ {entity} ?predicate ?object }}
            UNION
            {{ ?subject ?predicate {entity} }}
        }}
    """

    # Set the query and the return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and convert the results to a Python dictionary
    results = sparql.query().convert()

    list_of_triples = []

    results = results["results"]["bindings"]
    for result in results:
        triple = {}
        if "object" in result:
            triple["subject"] = entity
            triple["object"] = result["object"]["value"]
        else:
            triple["subject"] = result["subject"]["value"]
            triple["object"] = entity

        triple["predicate"] = result["predicate"]["value"]

        list_of_triples.append(triple)

    time.sleep(0.5)  # Sleep for 0.5 seconds to avoid timeouts
    return list_of_triples


data = read_json_(trainingdata_path)
processed_data = []

for data_block in tqdm(data[:10], desc="Processing data blocks"):
    list_of_tripples = []
    entities = data_block["author_dblp_uri"]
    if entities[0] == "[":
        entities = json.loads(entities.replace("'", '"'))[0]
        for key, entity in entities.items():
            tripples = retrieve_triples_for_entity(entity, endpoint_url)
            list_of_tripples += tripples
    else:
        list_of_tripples = retrieve_triples_for_entity(entities, endpoint_url)

    data_block["tripples_number"] = len(list_of_tripples)
    data_block["all_tripples"] = list_of_tripples
    processed_data.append(data_block)

# Ensure the processed data directory exists
os.makedirs(save_processed_data_path, exist_ok=True)

# Save the new processed training data
file_path = os.path.join(save_processed_data_path, "processed_data.json")
with open(file_path, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)
