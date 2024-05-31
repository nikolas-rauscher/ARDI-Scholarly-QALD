import json
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm 


trainindata_path = "data/external/trainingdata.js"
endpoint_url = "https://dblp-april24.skynet.coypu.org/sparql"  # SPARQL endpoint URL


def read_json_(trainindata_path) -> dict:
    with open(trainindata_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations


def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> dict:

    # Initialize the SPARQL wrapper with the endpoint URL
    sparql = SPARQLWrapper(endpoint_url)

    # SPARQL query
    query = f"""
            PREFIX dblp: <https://dblp.org/rdf/schema-2017-04-18#>

            SELECT ?predicate ?object
            WHERE {{
                {entity} ?predicate ?object.
            }}
            """


#   query = f"""
#   PREFIX dblp: <https://dblp.org/rdf/schema-2017-04-18#>
#
#   SELECT ?predicate ?object ?subject
#   WHERE {{
#       {{ {entity} ?predicate ?object }}
#    UNION
#       {{ ?subject ?predicate {entity} }}
#   }}
#   """

    # Set the query and the return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and convert the results to a Python dictionary
    results = sparql.query().convert()

    list_of_triples = []
    triple = {}
    for result in results["results"]["bindings"]:
        triple["subject"] = entity
        triple["predicate"] = result["predicate"]["value"]
        triple["object"] = result["object"]["value"]
        list_of_triples.append(triple)

    return list_of_triples


data = read_json_(trainindata_path)
qbar=tqdm(total=len(data))

for data_block in data:
    entities = data_block["author_dblp_uri"]
    if type(entities) == str:
        entities = [entities]
    for entity in entities:
        if (entity[0] == '['):
            L = json.loads(entity.replace('\'', '\"'))
            if (len(L) > 1):
                print("too many entities", entity, L)
                exit(-1)
            dic = L[0]
            for key, value in dic.items():
                res = retrieve_triples_for_entity(value, endpoint_url)
        else:
            res = retrieve_triples_for_entity(entity, endpoint_url)
    qbar.update(1)
