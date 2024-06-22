## How to process raw data

### All preedicatelabels that were found in the entire dtlb dataset (18):
```
'sameAs', 'wikidata', 'affiliation', 'full creator name', 'homepage URL', 'orcid', 'primary affiliation', 'primary full creator name', 'primary homepage URL', 'web page URL', 'wikipedia page URL', 'authored by', 'edited by', 'signature creator', 'creator note', 'archived page URL', 'differentFrom', 'award page URL'
```

### predicate explaination:

#### sameAs (can be removed)
The predicate sameAs used in the triple you provided is defined by the OWL (Web Ontology Language) and used within RDF (Resource Description Framework) data to indicate that two URI references actually refer to the same thing. That is, they are semantically equivalent.


#### wikidata (can be extended/removed)
The predicate wikidata in the triple you provided is used to link an entity from the DBLP database to its corresponding entry in Wikidata. 
Idea: retrieve also all tripples from wikidata with the given link 

#### affiliation (edit)
The affiliation predicate in the triple you provided is used to denote the institutional or organizational affiliation of an individual, in this case, within the context of the DBLP database. There is is no object label for the institution. The object can be used directly

#### full creator name (can be removed, since label already provides the full name)
The full creatorName predicate in the triple you provided, which is referenced as https://dblp.org/rdf/schema#creatorName, is designed to denote the full name of the creator, author, or originator of a work cataloged within the DBLP database.

#### homepage URL( can be extended/removed)
gives the homepage URL of the person 

#### orcid (can be removed)
ORCID provides a unique identifier for researchers, which helps resolve the common problem of name ambiguity in scholarly communications. By linking an ORCID ID with a researcher’s profile in DBLP, the database ensures that publications, citations, and other scholarly outputs are correctly attributed, even among researchers with similar or identical names.

#### primary affiliation (edit)
The primary affiliation predicate in the triple you provided is used to denote the primary institutional or organizational affiliation of an individual, in this case, within the context of the DBLP database. There is is no object label for the institution. The object can be used directly

#### primary full creator name (can be removed, since label already provides the full name)
The primary full creatorName predicate in the triple you provided is designed to denote the full name of the creator, author, or originator of a work cataloged within the DBLP database.

#### primary homepage URL (can be extended/removed)
gives the primary homepage URL of the person 

#### web page URL  (can be extended/removed)
gives the web page URL of the person 

#### wikipedia page URL (can be extended/removed)
gives the wikipedia page URL of the person 

#### authored by (keep)

#### edited by (keep)

#### signature creator (can be removed)
The signatureCreator predicate suggests a role that goes beyond typical authorship or contribution. It implies a level of creativity or responsibility that is pivotal or defining for the subject entity. This could be used in contexts where the specific input or creation by the individual is a distinguishing feature of the work or output.

#### creator note  (edit)
The creatorNote predicate in the triple you’ve provided is used to convey additional, often significant, information about an individual within the context of the DBLP database. There is is no object label for the institution. The object can be used directly

#### archived page URL (can be extended/removed)
gives archived page URL of the person 

#### differentFrom (keep)
The differentFrom predicate in the triple you’ve provided is used within RDF data to explicitly assert that two resources (entities) are distinct from each other, despite potentially similar names or attributes. This is particularly important in databases like DBLP, where multiple individuals may share similar or identical names. 

#### award page URL (can be extended/removed)
#### gives award page URL of the person 



#### All predicates, that do not have a label
['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://www.w3.org/2000/01/rdf-schema#label', 'http://purl.org/spar/datacite/hasIdentifier']

#### Type: http://www.w3.org/1999/02/22-rdf-syntax-ns#type (keep)
Used to specify the type or class of a resource, indicating what kind of thing a particular resource is, according to some ontology or schema.

#### Label: http://www.w3.org/2000/01/rdf-schema#label (can be removed)
used to provide a human-readable version of a resource’s name or title (label)

#### hasIdentifier: http://purl.org/spar/datacite/hasIdentifier (can be removed)
This predicate used for linking scholarly research to data repositories. This specific predicate is crucial for indicating that a given subject (such as a researcher or a publication) has an associated identifier, which in this case, is represented by an RDF node.


Ideas:
for google scholar link, use webscraping to get citations count
