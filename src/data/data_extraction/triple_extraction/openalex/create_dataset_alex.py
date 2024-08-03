import json
from .helper_function import run_query, read_json
import time



def create_alex_dataset(dblp_processed_dataset_path: str, outputdata_path: str, endpoint_url: str):
    """
    Retrieve all triples for the OpenAlex KG by taking Orcid from DBLP triples 

    Args:
        dblp_processed_dataset_path (str): The path to the processed DBLP dataset in JSON format.
        outputdata_path (str): The path where the augmented dataset should be saved.
        endpoint_url (str): URL of the SPARQL endpoint to fetch triples.

    Outputs:
        A new JSON file with the retrieved triples
    """
    print("Extracting triples for OpenAlex KG...\n")
    data = read_json(dblp_processed_dataset_path)
    alex_all_questions = []

    for question, iteration in zip(data, range(len(data))):
        time.sleep(1)
        print(iteration,"/",len(data))
        alex_question = {}
        alex_question["id"] = question["id"]
        alex_question["question"] = question["question"]
        alex_question["answer"] = question["answer"]
        alex_question["author_uri"] = []
        alex_question["triples_number"] = 0
        alex_question["all_triples"] = []

        orcid = ""
        for dblp_entity in question["all_triples"]: #go over every entity in quesiton 
            entity_dict = {}
            for triple in dblp_entity["triples"]:
                if triple["predicateLabel"] == "orcid":
                    orcid = triple["object"]
                    entity_dict["entity"] = orcid

            query = f"""
                PREFIX dbo: <https://dbpedia.org/ontology/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?author ?name
                    WHERE {{
                    ?author foaf:name ?name .
                    ?author dbo:orcidId "{orcid}"^^<http://www.w3.org/2001/XMLSchema#string> .
                }}
            """
            query_res = run_query(query, endpoint_url)
            triples = []
            if query_res['results']['bindings']:
                author_url_alex = query_res['results']['bindings'][0]['author']['value']
                alex_question["author_uri"].append(author_url_alex)
                author_name = query_res['results']['bindings'][0]['name']['value']

                query = f"""
                    SELECT ?predicate ?object
                    WHERE {{
                        <{author_url_alex}>  ?predicate ?object.
                    }}
                    """
                query_res = run_query(query, endpoint_url)

                for binding in query_res['results']['bindings']:
                    entry = {
                        "subject": author_name,
                        "predicate": (binding['predicate']['value']).split('/')[-1] ,
                        "object": binding['object']['value']
                    }
                    triples.append(entry)

                query = f"""
                    SELECT ?subject ?predicate ?object 
                    WHERE {{
                        ?subject  ?predicate <{author_url_alex}>.
                    }}
                    """
                query_res = run_query(query, endpoint_url)

                for binding in query_res['results']['bindings']:
                    entry = {
                        "subject": binding['subject']['value'],
                        "predicate": (binding['predicate']['value']).split('/')[-1], 
                        "object": author_name
                    }
                    triples.append(entry)

                entity_dict["triples"] = triples
                alex_question["all_triples"].append(entity_dict)
                triples_number = 0
                for entity in alex_question["all_triples"]:
                    triples_number += len(entity["triples"])
                alex_question["triples_number"] = triples_number
            
        alex_all_questions.append(alex_question)

        with open(outputdata_path, 'w') as file:
            json.dump(alex_all_questions, file, indent=4,ensure_ascii=False)
        
    print("Finished extracting triples for OpenAlex KG\n")


##############################################################################
def main():
    """
    To run this script direcly run:
        python -m src.data.data_extraction.triple_extraction.openalex.create_dataset_alex   
    from the root directory of this project 
    """
    dblp_processed_dataset_path = "data/interim/dblp/pre_processed_data10.json"
    outputdata_path = "data/interim/alex/pre_processed_data10.json"
    endpoint_url = "https://semopenalex.org/sparql" # Alternative smaller KG: "https://semoa.skynet.coypu.org/sparql"

    create_alex_dataset(dblp_processed_dataset_path, outputdata_path, endpoint_url)
##############################################################################
if __name__ == "__main__":
    main()