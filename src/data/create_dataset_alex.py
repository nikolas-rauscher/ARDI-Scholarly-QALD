# get all orcids of every author
# get author object for that orcid 
# get all tripples for that author object 

import json
from run_query import run_query


def get_data() -> dict : 
    outputdata_name_path_0 = "data/processed/processed_data3_0.json"
    outputdata_name_path_1 = "data/processed/processed_data3_1.json"

    def read_json_(outputdata_name_path) -> dict:
        with open(outputdata_name_path, 'r', encoding='utf-8') as file:
            formulations = json.load(file)
        return formulations

    data_0 = read_json_(outputdata_name_path_0)
    data_1 = read_json_(outputdata_name_path_1)

    data = data_0 + data_1 
    return data

alex_all_questions = []

data = get_data() 
data = data[:3000]
for question, iteration in zip(data, range(len(data))):
    print(iteration)
    alex_question = {}
    alex_question["id"] = question["id"]
    alex_question["question"] = question["question"]
    alex_question["answer"] = question["answer"]

    orcid = ""
    for tripple in question["all_tripples"]:
        if tripple["predicateLabel"] == "orcid":
            orcid = tripple["object"]

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
        alex_question["author_uri"] = author_url_alex

        author_name = query_res['results']['bindings'][0]['name']['value']

    #{'head': {'vars': ['author', 'name']}, 'results': {'bindings': [{'author': {'type': 'uri', 'value': 'https://semopenalex.org/author/A5069855349'}, 'name': {'type': 'literal', 'value': 'Tom Wilson'}}]}}
    query = f"""

        SELECT ?predicate ?object
        WHERE {{
            <{author_url_alex}>  ?predicate ?object.
        }}

        """
    query_res = run_query(query)

    tripples = []
    # Iterate over each binding in the data
    for binding in query_res['results']['bindings']:
        # Each entry in the list is a dictionary with 'Subject', 'Predicate', 'Object'
        entry = {
            "subject": author_name,
            "predicate": (binding['predicate']['value']).split('/')[-1] ,
            "object": binding['object']['value']
        }
        tripples.append(entry)

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
        tripples.append(entry)

    alex_question["tripples_number"] = len(tripples)

    alex_question["all_tripples"] = tripples

    alex_all_questions.append(alex_question)

file_path = "data/processed/processed_data_alex_01.json"
with open(file_path, 'w') as file:
    json.dump(alex_all_questions, file, indent=4,ensure_ascii=False)



##############################################################################
def main():

   get_data()
##############################################################################
if __name__ == "__main__":
    main()