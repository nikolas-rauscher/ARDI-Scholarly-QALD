
#retrieve and count all predicates by their occurence

"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?predicate (COUNT(?subject) AS ?count)
        WHERE {{ ?subject ?predicate ?object.}}
        GROUP BY ?predicate
        ORDER BY DESC(?count)
        LIMIT 100
"""
    

            



