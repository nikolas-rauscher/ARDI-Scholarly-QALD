from SPARQLWrapper import SPARQLWrapper, JSON
import json

#endpoint_url ="https://semopenalex.org/sparql"
endpoint_url ="https://semoa.skynet.coypu.org/sparql"#"" #SPARQL endpoint URL


def read_json(outputdata_name_path: str) -> dict:
    """
    Parameters:
    - trainindata_path (str): The file path to the JSON file to be read.

    Returns:
    - formulations (List[Dict]): A list of dictionaries parsed from the JSON file. 
    """
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations


def save_intermediate_result(outputdata_path: str, new_dataset: dict) -> None:
    """
    Args:
        outputdata_path (str): The file path where the dataset should be saved.
        new_dataset (dict): The dataset to be saved
    """
    with open(outputdata_path, 'w', encoding='utf-8') as file:
        json.dump(new_dataset, file, indent=4, ensure_ascii=False)
        

def run_query(query: str, endpoint_url: str) -> dict:
    """
    Executes a SPARQL query against a specified endpoint and returns the results.

    Args:
        query (str): The SPARQL query to be executed.
        endpoint_url (str): The URL of the SPARQL endpoint.

    Returns:
        dict: A dictionary containing the results of the SPARQL query.
   """
    sparql = SPARQLWrapper(endpoint_url) #Initialize the SPARQL wrapper with the endpoint URL    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


  





