import json
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict, Optional

def read_json(trainindata_path: str, subset: Optional[int] = None) -> List[Dict]:
    """
    Reads a JSON file and optionally returns a subset of its contents.

    Parameters:
    - trainindata_path (str): The file path to the JSON file to be read.
    - subset (Optional[int]): The number of items to return from the top of the JSON data. If None, the entire data is returned.

    Returns:
    - formulations (List[Dict]): A list of dictionaries parsed from the JSON file. 
    """
    with open(trainindata_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)

    if subset is not None:
        formulations = formulations[:subset]

    return formulations


from typing import List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON

def get_triples_for_entity(query: str, endpoint_url: str, placing: str, entity: str) -> List[Dict[str, str]]:
    """
    Executes a SPARQL query to retrieve triples from a specified endpoint and organizes the results into a list of dictionaries.

    Parameters:
    - query (str): The SPARQL query string to be executed.
    - endpoint_url (str): The URL of the SPARQL endpoint where the query should be run.
    - placing (str): Specifies whether the `entity` is used as a 'subject' or an 'object' in the triples.
    - entity (str): The URI of the entity to be inserted into the triple's 'subject' or 'object', depending on the 'placing'.

    Returns:
    - list_of_triples (List[Dict[str, str]]): A list of dictionaries, each representing a triple with keys 'subject', 'predicate', 'object', and optional label keys 'subjectLabel', 'predicateLabel', 'objectLabel'. Each key maps to its corresponding value in the triple.

    Each triple dictionary contains:
    - 'subject' (str): The subject of the triple.
    - 'predicate' (str): The predicate of the triple.
    - 'object' (str): The object of the triple.
    - 'subjectLabel' (str, optional): The label of the subject, if available.
    - 'predicateLabel' (str, optional): The label of the predicate, if available.
    - 'objectLabel' (str, optional): The label of the object, if available.
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    list_of_triples = []
    for result in results["results"]["bindings"]:
        triple = {}
        if placing == "subject":
            triple["subject"] = entity
            triple["object"] = result["object"]["value"]
        elif placing == "object":
            triple["object"] = entity
            triple["subject"] = result["subject"]["value"]

        triple["predicate"] = result["predicate"]["value"]
        triple["subjectLabel"] = result.get("subjectLabel", {}).get("value", "")
        triple["predicateLabel"] = result.get("predicateLabel", {}).get("value", "")
        triple["objectLabel"] = result.get("objectLabel", {}).get("value", "")
        list_of_triples.append(triple)

    return list_of_triples