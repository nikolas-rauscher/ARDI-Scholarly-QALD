"""Script to download or generate data"""

"""Script to download or generate data"""

import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON


save_processed_data_path = "data/processed"
endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL


def retrieve_triples_for_entity(entity: str, endpoint_url: str) -> list:

    sparql = SPARQLWrapper(endpoint_url) #Initialize the SPARQL wrapper with the endpoint URL

    #SPARQL query
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?predicate (COUNT(?subject) AS ?count)
        WHERE {{ ?subject ?predicate ?object.}}
        GROUP BY ?predicate
        ORDER BY DESC(?count)
        LIMIT 100
    """
    
    # Set the query and the return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and convert the results to a Python dictionary
    results = sparql.query().convert()
    print(results["results"]["bindings"])

   


retrieve_triples_for_entity("entity",endpoint_url)
 

    
    
            



