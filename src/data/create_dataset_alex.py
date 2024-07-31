# get all orcids of every author
# get author object for that orcid 
# get all triples for that author object 

import json
from run_query import run_query


def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

def create_alex_dataset(dblp_processed_dataset_path, outputdata_path):
    data = read_json_(dblp_processed_dataset_path) 

    alex_all_questions = []

    for question, iteration in zip(data, range(len(data))):
        print(iteration)
        alex_question = {}
        alex_question["id"] = question["id"]
        alex_question["question"] = question["question"]
        alex_question["answer"] = question["answer"]
        alex_question["author_uri"] = []
        alex_question["all_triples"] = []

        orcid = ""
        for dblp_entity in question["all_triples"]: #go over every entity in quesiton 
            entity_dict = {}
            for tripple in dblp_entity["triples"]:
                if tripple["predicateLabel"] == "orcid":
                    orcid = tripple["object"]
                    entity_dict["entity"] = orcid
                


            query = f"""
                PREFIX dbo: <https://dbpedia.org/ontology/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>

                SELECT ?author ?name
                    WHERE {{
                    
                    ?author foaf:name ?name .
                    ?author dbo:orcidId "{orcid}" .
            
                }}
            """
            query_res = run_query(query)  
            if query_res['results']['bindings']:

                author_url_alex = query_res['results']['bindings'][0]['author']['value']
                #alex_question["author_uri"] =  author_url_alex
                alex_question["author_uri"].append(author_url_alex)

                author_name = query_res['results']['bindings'][0]['name']['value']

            #{'head': {'vars': ['author', 'name']}, 'results': {'bindings': [{'author': {'type': 'uri', 'value': 'https://semopenalex.org/author/A5069855349'}, 'name': {'type': 'literal', 'value': 'Tom Wilson'}}]}}
            query = f"""

                SELECT ?predicate ?object
                WHERE {{
                    <{author_url_alex}>  ?predicate ?object.
                }}

                """
            query_res = run_query(query)

            triples = []
            # Iterate over each binding in the data
            for binding in query_res['results']['bindings']:
                # Each entry in the list is a dictionary with 'Subject', 'Predicate', 'Object'
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
            query_res = run_query(query)

            # Iterate over each binding in the data
            for binding in query_res['results']['bindings']:
                # Each entry in the list is a dictionary with 'Subject', 'Predicate', 'Object'
                entry = {
                    "subject": binding['subject']['value'],
                    "predicate": (binding['predicate']['value']).split('/')[-1], 
                    "object": author_name
                }
                triples.append(entry)


            entity_dict["triples"] = triples

            alex_question["all_triples"].append(entity_dict)

            for entity in alex_question["all_triples"]:
                triples_number = 0
                triples_number += len(entity["triples"])
            alex_question["triples_number"] = triples_number
            
        alex_all_questions.append(alex_question)

    with open(outputdata_path, 'w') as file:
        json.dump(alex_all_questions, file, indent=4,ensure_ascii=False)



##############################################################################
def main():

    dblp_processed_dataset_path = "data/processed/dblp/pre_processed_data1000.json"
    outputdata_path = "data/processed/alex/pre_processed_data1000.json"
    create_alex_dataset(dblp_processed_dataset_path, outputdata_path)
##############################################################################
if __name__ == "__main__":
    main()