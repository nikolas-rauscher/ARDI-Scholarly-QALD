# Dataset creation

## Overview

All scripts in this folder are designed to create a final coherent dataset with all nessary information in order to answer the question from the provided data given one or more dblp URIs of the authors. 

## Order in which to generate the final dataset:
``` 
create_dataset_dblp --> postprocessing_dblp-----------------------------|
                                                                        |
                                                                        ------->merge_triples
                                                                        |
create_dataset_alex   --> postprocessing_alex --|                       |
                                                |                       |
                                                |---->add_wiki----------|
                                                |
process_wiki -----------------------------------|  
``` 
### Provided dataset with questions:
```python
[
    {
        "id": "d086b012-ca96-402f-856d-afce74fce8fc",
        "question": "How many times have publications from the institution of the academic who authored the paper 'Managing customer oriented research' been cited?",
        "author_dblp_uri": "<https://dblp.org/pid/94/11523>"
    },
    {
        "id": "d3aa86e9-f6f8-47a4-821b-6bf35d68415d",
        "question": "Where was the author of the publication 'The taming of the shrew: increasing performance by automatic parameter tuning for java garbage collectors' born?",
        "author_dblp_uri": "<https://dblp.org/pid/m/HMossenbock>"
    }..

]
```

#### create_dataset_dblp.py

This script creates a list of dictionaries of all extracted triples from the dblp Knowledge graph given the Author URI. 

**Input:**<br>
Path of dataset with the questions <br>
**Output:** <br>
Json file with all extracted triples for every question 


#### postprocessing_dblp.py

This script postprocess all extracted triples from the DBLP KG, by removing unnecessary triples and replacing the names of certain predicates

**Input**: <br>
Path of extracted triples from DBLP KG <br>
**Output**: <br>
Json of all processed triples, extracted from the DBLP KG

```python
[
    {
        "id": "6b8aa79c-3908-4f03-b85b-aa1a325d9fe6",
        "question": "What type of information sources were found to be lacking in organized information at Social Services offices according to the author's observation?",
        "answer": "oral communication and notes",
        "author_dblp_uri": "<https://dblp.org/pid/w/TDWilson>",
        "triples_number": 885,
        "all_triples": [
            {
                "entity": "<https://dblp.org/pid/w/TDWilson>",
                "triples": [
                    {
                        "subject": "<https://dblp.org/pid/w/TDWilson>",
                        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        "object": "https://dblp.org/rdf/schema#Creator",
                        "subjectLabel": "Thomas D. Wilson 0001",
                        "predicateLabel": "",
                        "objectLabel": "Creator"
                    }..
                ]..
            }   
        ]..
    }..
]
```

#### create_dataset_alex.py

This script creates a list of dictionaries of all extracted triples from the OpenAlex Knowledge Graph given the DBLP Author URI. First the Author ORCID is extracted from the extracted triples from the DBLP KG. Then the Author URI in the OpenAlex dataset is extracted by providing the Orcid of the author


**Input:** <br>
Path of dataset with the questions <br>
Path of extracted triples from DBLP dataset <br>
**Output:** <br> 
Json file with all extracted triples from OpenAlex KG for every question 


```python
[
    {
        "id": "6b8aa79c-3908-4f03-b85b-aa1a325d9fe6",
        "question": "What type of information sources were found to be lacking in organized information at Social Services offices according to the author's observation?",
        "answer": "oral communication and notes",
        "author_uri": [
            "https://semopenalex.org/author/A5069855349"
        ],
        "all_triples": [
            {
                "entity": "https://orcid.org/0000-0003-4941-8443",
                "triples": [
                    {
                        "subject": "Tom Wilson",
                        "predicate": "22-rdf-syntax-ns#type",
                        "object": "https://semopenalex.org/ontology/Author"
                    }...
                ]...
            }..
        ]..
    }..
]
```

#### postprocessing_alex.py

This script postprocess all extracted triples from the OpenAlex KG, by 
- Predicate Filtering: The script filters triples based on a list of desired predicates such as 'modified', 'citedByCount', etc. Only triples with these predicates are retained for further processing.
- Type Transformation: If a triple’s predicate is "22-rdf-syntax-ns#type", the script simplifies it to "is", and modifies the object to only include the last segment of the URI, making the triple more readable.
- Creator Information: When the predicate is "creator", the script retrieves the title of the work associated with this predicate via a SPARQL query, and replaces the triple with a more descriptive version indicating that the work was written by the specified entity.
- Authorship Details: For "hasAuthor" predicates, the script queries for affiliation information and the title of the work. It then creates a new triple stating that the author was working at the specified affiliations while writing the paper, providing contextual information about the author’s role and environment.
- Membership Information: If the predicate is "org#memberOf", the script fetches the name of the institution from the URI and constructs a triple indicating that the subject is a member of the fetched institution.
- Counts by Year: For the "countsByYear" predicate, the script fetches the citation counts and works count for the specified year, transforming these into more explicit triples that clearly state the number of citations and papers in that particular year.

Input: Path of extracted triples from DBLP KG
Output: list of transformed and enriched triples, as well as a count of these triples. This processed data is saved in a structured JSON format which retains the enriched information in an organized manner. 

```python
[
    {
        "id": "6b8aa79c-3908-4f03-b85b-aa1a325d9fe6",
        "question": "What type of information sources were found to be lacking in organized information at Social Services offices according to the author's observation?",
        "answer": "oral communication and notes",
        "author_uri": [
            "https://semopenalex.org/author/A5069855349"
        ],
        "all_triples": [
            {
                "entity": "https://orcid.org/0000-0003-4941-8443",
                "triples": [
                    {
                        "subject": "Tom Wilson",
                        "predicate": "22-rdf-syntax-ns#type",
                        "object": "Author"
                    }...
                ]...
            }..
        ]..
    }..
]
```

#### process_wikidata.py

This script process the partially corrupted wiki dataset, that was provided by the organizers and creates a Json file with the data

**Input:** <br>
Path of corrupted wikidata as text file <br>
**Ouput:** <br> 
Well formated Json file with with wiki data 

#### add_wiki_data.py

For each extracted name and institution, the script attempts to find matching Wikipedia articles using two key functions:
- find_wiki_article_by_name(): Searches for Wikipedia articles where the entity’s name appears to match an entity within the article’s initial text, using a cosine similarity measure on character n-grams.
- find_wiki_article_by_institution(): Searches for articles where the institution’s name appears to match entities within the article’s text, again using cosine similarity on character n-grams.
- These searches are designed to identify relevant Wikipedia articles based on textual similarity using cusine similarity between the article content and the names or institutions related to the academic entity.

Input: Path of the processed triples from OpenAlex KG 
Ouput: Processed triples from OpenAlex KG plus the relevant wikipedia data


```python
[
    {
        "id": "6b8aa79c-3908-4f03-b85b-aa1a325d9fe6",
        "question": "What type of information sources were found to be lacking in organized information at Social Services offices according to the author's observation?",
        "answer": "oral communication and notes",
        "triples_number": 574,
        "author_uri": [
            "https://semopenalex.org/author/A5069855349"
        ],
        "all_triples": [
            {
                "entity": "https://orcid.org/0000-0003-4941-8443",
                "triples": [
                    {
                        "subject": "Tom Wilson",
                        "predicate": "is",
                        "object": "Author"
                    }...
                ]..
            }
        ]..


        "wiki_data": [
                    {
                        "author_wikipedia_text": "Dr. Thomas D. Wilson is a .."
                    }..
        ]
    }..
]
```
                        
### merge_triples.py

combines the extracted triples from the DBLP dataset and the ApenAlex dataset and wikidata to one final JSON dataset.

**Input:** <br>
- Path of the processed triples from DBLP KG  <br>
- Path Processed triples from OpenAlex KG plus the relevant wikipedia data

**Output:** <br>
- final dataset

