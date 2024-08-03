#get all destinct entities in DBLP KG
"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?type
WHERE {
  ?entity rdf:type ?type .
}"""


#get the amount of every entity in the DBLP KG
"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?type (COUNT(?entity) AS ?count)
WHERE {
  ?entity rdf:type ?type .
}
GROUP BY ?type
ORDER BY DESC(?count)

"""


#get the amount of all entities together in DBLP KG 
"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT (COUNT(DISTINCT ?type) AS ?typeCount)
WHERE {
  ?entity rdf:type ?type .
}
"""


#get all destinct predicates in DBLP
"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?predicate
WHERE {
  ?subject ?predicate ?object .
}
"""

#get the amoount of all destinct predicates in DBLP
"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT (COUNT(DISTINCT ?predicate) AS ?numberOfPredicates)
WHERE {
  ?subject ?predicate ?object .
}
"""
