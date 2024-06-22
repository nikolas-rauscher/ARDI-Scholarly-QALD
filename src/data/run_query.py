from SPARQLWrapper import SPARQLWrapper, JSON


save_processed_data_path = "data/processed"
endpoint_url ="https://semopenalex.org/sparql" #SPARQL endpoint URL



def run_query(query):
    sparql = SPARQLWrapper(endpoint_url) #Initialize the SPARQL wrapper with the endpoint URL

    
    # Set the query and the return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)


    # Execute the query and convert the results to a Python dictionary
    results = sparql.query().convert()
    return results


query= query = f"""
                    PREFIX soa: <https://semopenalex.org/ontology/>
                    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                    PREFIX dcterms: <http://purl.org/dc/terms/>

                    SELECT ?affiliation ?paper
                    WHERE {{
                      <https://semopenalex.org/authorship/W1951090766A5069855349> soa:rawAffiliation ?affiliation .
                      ?work soa:hasAuthorship <https://semopenalex.org/authorship/W1951090766A5069855349> .
                      ?work dcterms:title ?paper
                    }}
                    """

query_res= run_query(query)
affiliation_list = []
#print(query_res)
paper_during_affiliation = query_res['results']['bindings'][0]['paper']['value']
for i in range(len(query_res['results']['bindings'])):
              affiliation_list.append(query_res['results']['bindings'][i]['affiliation']['value'])
print(paper_during_affiliation)
print(affiliation_list)   



