import json
from run_query import run_query


predicates = ['modified', 'citedByCount', 'worksCount', 'h-index', 'name', '2YrMeanCitedness', 'i10Index', 'alternativeName', 'orcidId', 'scopus', 'twitter']
predicates_process = ["creator", "countsByYear", "org#memberOf", "hasAuthor", "22-rdf-syntax-ns#type" ]


def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations





data = read_json_("data/processed/alex/processed_data_alex_01.json")
data = data[:10]
new_dataset =  []
for question, i in zip(data, range(len(data))):
    print(i)
    new_question = {}
    new_question["id"] = question["id"]
    new_question["question"] = question["question"]
    new_question["answer"] = question["answer"]
    if ("author_uri" in question): new_question["author_uri"] =  question["author_uri"]  #TODO quick fix
 
    
    new_tripples = []
    for tripple in question["all_tripples"]:
        new_tripple = {}


        if tripple["predicate"] in predicates:
            new_tripples.append(tripple)

        if tripple["predicate"] == "22-rdf-syntax-ns#type": 
            new_tripple["subject"] = tripple["subject"]
            new_tripple["predicate"]= "is"
            new_tripple["object"]= tripple["object"].split('/')[-1]
            new_tripples.append(new_tripple)

        if tripple["predicate"] == "creator":
          title_uri = tripple["subject"]
          query = f"""
                  PREFIX dcterms: <http://purl.org/dc/terms/>

                  SELECT ?title
                  WHERE {{
                    <{title_uri}> dcterms:title ?title .
                  }}
                  """
          query_res = run_query(query)
          if query_res['results']['bindings']:
            paper_title = query_res['results']['bindings'][0]['title']['value']
            #print("paper_title:",paper_title)
            new_tripple["subject"] = paper_title
            new_tripple["predicate"]= "written by"
            new_tripple["object"]= tripple["object"]

            new_tripples.append(new_tripple)
          else:
             print("no title found") 
          


        if tripple["predicate"] == "hasAuthor":
            authorship_uri = tripple["subject"]
            query = f"""
                    PREFIX soa: <https://semopenalex.org/ontology/>
                    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                    PREFIX dcterms: <http://purl.org/dc/terms/>

                    SELECT ?affiliation ?paper
                    WHERE {{
                      <{authorship_uri}> soa:rawAffiliation ?affiliation .
                      ?work soa:hasAuthorship <{authorship_uri}> .
                      ?work dcterms:title ?paper
                    }}
                    """
                
            query_res = run_query(query)
            affiliation_list = []
             
            for i in range(len(query_res['results']['bindings'])):
              affiliation_list.append(query_res['results']['bindings'][i]['affiliation']['value'])


            if affiliation_list:
              # only if there are affiliations get the paper that was written during those affiliations 
              paper_during_affiliation = query_res['results']['bindings'][0]['paper']['value']
              #print("affiliation_list:",affiliation_list)


              new_tripple["subject"] =  tripple["object"] 
              new_tripple["predicate"]= "was working in"
              new_tripple["object"]= ", ".join(affiliation_list) + " while writing paper: " + paper_during_affiliation

              new_tripples.append(new_tripple)
            else:
               pass # remove tripple
            

        if tripple["predicate"] == "org#memberOf":
              institutaion_uri = tripple["object"]
              query = f"""
                      PREFIX foaf: <http://xmlns.com/foaf/0.1/>

                      SELECT ?instituation_name
                        WHERE {{
                          <{institutaion_uri}> foaf:name ?instituation_name .
                        }}
                      """
              query_res = run_query(query)
              institutation_list = []
      
              for i in range(len(query_res['results']['bindings'])):
                institutation_list.append(query_res['results']['bindings'][i]['instituation_name']['value'])
              #print("institutation_list:",institutation_list)

              if institutation_list:
                new_tripple["subject"] =  tripple["subject"] 
                new_tripple["predicate"]= "member of"
                new_tripple["object"]= institutation_list

                new_tripples.append(new_tripple)
              else:
                pass # remove tripple  

        

        if tripple["predicate"] == "countsByYear":
              counts_by_year_uri = tripple["object"]
              query = f"""
                PREFIX soa: <https://semopenalex.org/ontology/>
                SELECT ?citations_counts ?works_count
                        WHERE {{
                            <{counts_by_year_uri}> soa:citedByCount ?citations_counts .
                            <{counts_by_year_uri}> soa:worksCount ?works_count .
                              }}
                """
              query_res = run_query(query)
              
              if query_res['results']['bindings'][0]:
                counts_by_year = (query_res['results']['bindings'][0]['citations_counts']['value'])
                
                #print("counts_by_year:",counts_by_year)
                new_tripple["subject"] =  tripple["subject"] 
                new_tripple["predicate"]= "citatations in year " + counts_by_year_uri[-4:]
                new_tripple["object"]= counts_by_year
                new_tripples.append(new_tripple)

                works_count = (query_res['results']['bindings'][0]['works_count']['value'])
                #print("works_count:",works_count)
                new_tripple["subject"] =  tripple["subject"] 
                new_tripple["predicate"]= "works in year " + counts_by_year_uri[-4:]
                new_tripple["object"]= works_count
                new_tripples.append(new_tripple)

              else:
                pass # remove tripple  
    new_question["tripples_number"] = len(new_tripples)
    new_question["all_tripples"] = new_tripples
    new_dataset.append(new_question)  



file_path = "data/processed/alex/processed_alex_first_30.json"

with open(file_path, 'w') as file:
    json.dump(new_dataset, file, indent=4,ensure_ascii=False)
