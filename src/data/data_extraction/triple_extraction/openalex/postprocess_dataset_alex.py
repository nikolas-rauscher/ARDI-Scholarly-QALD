import json
import os
from .helper_function import run_query, read_json, save_intermediate_result
import concurrent.futures
import time

predicates = ['modified', 'citedByCount', 'worksCount', 'h-index', 'name', '2YrMeanCitedness', 'i10Index', 'alternativeName', 'orcidId', 'scopus', 'twitter']
predicates_process = ["creator", "countsByYear", "org#memberOf", "hasAuthor", "22-rdf-syntax-ns#type" ]


def post_process_alex(data_prepocessed: dict, outputdata_path: str, endpoint_url: str) -> None:
  """
    
    This function transforms the retieved triples from OpenAlex based on predefined rules, queries additional information from a SPARQL endpoint,
    and compiles a new dataset with enriched triples. The function iterates each triple,
    performing specific transformations or queries based on the triple's predicate.

    Args:
        pre_processed_data_name (str): The filename of the pre-processed dataset to load.
        outputdata_name (str): The filename for saving the post-processed dataset.
        endpoint_url (str): The URL of the SPARQL endpoint to query additional data.
  """
  data = data_prepocessed
  new_dataset =  []
  for question, i in zip(data, range(len(data))):
      print(i,"/",len(data))
      new_question = {}
      new_question["id"] = question["id"]
      new_question["question"] = question["question"]
      new_question["answer"] = question["answer"]
      new_question["tripples_number"] = 0
      if ("author_uri" in question): new_question["author_uri"] =  question["author_uri"] 
      new_question["all_triples"] = []
        
      for entity in question["all_triples"]:
        entity_dict = {}
        new_tripples = []
        for triple in entity["triples"]:
          new_tripple = {}


          if triple["predicate"] in predicates:
              new_tripples.append(triple)

          if triple["predicate"] == "22-rdf-syntax-ns#type": 
              new_tripple["subject"] = triple["subject"]
              new_tripple["predicate"]= "is"
              new_tripple["object"]= triple["object"].split('/')[-1]
              new_tripples.append(new_tripple)

          if triple["predicate"] == "creator":
            title_uri = triple["subject"]
            query = f"""
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    SELECT ?title
                    WHERE {{
                      <{title_uri}> dcterms:title ?title .
                    }}
                    """
            query_res = run_query(query,endpoint_url)
            if query_res['results']['bindings']:
              paper_title = query_res['results']['bindings'][0]['title']['value']
              new_tripple["subject"] = paper_title
              new_tripple["predicate"]= "written by"
              new_tripple["object"]= triple["object"]
              new_tripples.append(new_tripple)
            
          if triple["predicate"] == "hasAuthor":
              authorship_uri = triple["subject"]
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
                
              query_res = run_query(query,endpoint_url)
              affiliation_list = []
              
              for i in range(len(query_res['results']['bindings'])):
                affiliation_list.append(query_res['results']['bindings'][i]['affiliation']['value'])

              if affiliation_list:
                # only if there are affiliations get the paper that was written during those affiliations 
                paper_during_affiliation = query_res['results']['bindings'][0]['paper']['value']
                new_tripple["subject"] =  triple["object"] 
                new_tripple["predicate"]= "was working in"
                new_tripple["object"]= ", ".join(affiliation_list) + " while writing paper: " + paper_during_affiliation
                new_tripples.append(new_tripple)
              
          if triple["predicate"] == "org#memberOf":
                institutaion_uri = triple["object"]
                query = f"""
                        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                        SELECT ?instituation_name
                          WHERE {{
                            <{institutaion_uri}> foaf:name ?instituation_name .
                          }}
                        """
                query_res = run_query(query,endpoint_url)
                institutation_list = []
        
                for i in range(len(query_res['results']['bindings'])):
                  institutation_list.append(query_res['results']['bindings'][i]['instituation_name']['value'])

                if institutation_list:
                  new_tripple["subject"] =  triple["subject"] 
                  new_tripple["predicate"]= "member of"
                  new_tripple["object"]= institutation_list
                  new_tripples.append(new_tripple)

          if triple["predicate"] == "countsByYear":
                counts_by_year_uri = triple["object"]
                query = f"""
                  PREFIX soa: <https://semopenalex.org/ontology/>
                  SELECT ?citations_counts ?works_count
                          WHERE {{
                              <{counts_by_year_uri}> soa:citedByCount ?citations_counts .
                              <{counts_by_year_uri}> soa:worksCount ?works_count .
                                }}
                  """
                query_res = run_query(query,endpoint_url)
                
                if query_res['results']['bindings']:
                  counts_by_year = (query_res['results']['bindings'][0]['citations_counts']['value'])
                  year = counts_by_year_uri[-4:] 
                  new_tripple["subject"] =  triple["subject"] 
                  new_tripple["predicate"]= "citations in year " + year 
                  new_tripple["object"]= counts_by_year
                  new_tripples.append(new_tripple)
                  works_count = (query_res['results']['bindings'][0]['works_count']['value'])
                  new_tripple["subject"] =  triple["subject"] 
                  new_tripple["predicate"]= " written paper amount in year" + " " + year + ":"
                  new_tripple["object"]= works_count
                  new_tripples.append(new_tripple)

        entity_dict["entity"] = entity["entity"]
        entity_dict["triples"] = new_tripples
        new_question["all_triples"].append(entity_dict)
      
      for entity in new_question["all_triples"]:
          triples_number = 0
          triples_number += len(entity["triples"])
      new_question["triples_number"] = triples_number
      new_dataset.append(new_question)  

      save_intermediate_result(outputdata_path, new_dataset)



def process_in_parallel(data: dict, outputdata_path: str, endpoint_url: str, processes: int) -> list:
    """
    Splits the data into segments and processes each segment in parallel.

    Parameters:
    - data (list): The data to be processed.
    - outputdata_name (str): The base name for output files for each segment.
    - endpoint_url (str): URL to the SPARQL endpoint for data processing.
    - processes (int): Number of parallel processes to execute.

    Returns:
    - outputnames (list): A list of filenames where the processed data segments are stored.
    """
    data_length = len(data)
    segment_size = data_length // processes  # Calculate segment size
    outputpathes = []
    data_segments = []
    if data_length < processes:
       processes = 1
       print("warning: there are less questions then processes. Process number is reduced..")
    # Create segments and corresponding output names
    print(f"parallelize postprocessing on {processes} processes")
    for i in range(processes):
        outputdata_name  = os.path.basename(outputdata_path)
        directory = os.path.dirname(outputdata_path)
        print(outputdata_name)
        outputpath = directory + "/" +f"{outputdata_name}_{i}.json"
        start_index = i * segment_size
        if i == processes - 1:
            data_segments.append(data[start_index:]) 
        else:
            end_index = start_index + segment_size
            data_segments.append(data[start_index:end_index])
        outputpathes.append(outputpath)
    # Execute processing in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(post_process_alex, data_segments[i], outputpathes[i], endpoint_url) for i in range(processes)]
        for future in concurrent.futures.as_completed(futures):
            print(f"One process finished")  
    return outputpathes


def merge_and_save(outputpathes: list, final_output_name: str):
    """
    Merges data from multiple segment files into a single file and saves it.

    Parameters:
    - outputnames (list): A list of filenames containing the data segments to be merged.
    - final_output_name (str): The final output filename where the merged data will be saved.
    """
    merged_data = []
    for filepath in outputpathes:
        with open(filepath, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)
    with open(final_output_name, 'w') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    print(f"Data merged and saved to {final_output_name}")
  
    
def post_process_alex_parallelized(data_alex_prepocessed_path: str, outputdata_path: str, endpoint_url: str, processes = 1, delete_intermediate = True): 
    """
    Manages the parallel processing of data for post-processing from a specified dataset.

    Parameters:
    - pre_processed_data_name (str): Filename of the pre-processed data.
    - outputdata_name (str): Filename for the final processed output.
    - endpoint_url (str): URL of the SPARQL endpoint for querying additional data.
    - processes (int): Number of processes to use for parallel execution.
    - delete_intermediate (bool): Flag to indicate whether to delete intermediate files.

    Executes parallel processing if processes > 1, otherwise processes data serially.
    Merges all data into one final file and optionally deletes intermediate files.
    """
    print("Postprocessing triples for OpenAlex KG...\n")
    data_alex_prepocessed = read_json(data_alex_prepocessed_path)
    if processes ==1:
        post_process_alex(data_alex_prepocessed, outputdata_path, endpoint_url)
    elif processes>1 and type(processes) == int:
        outputnames = process_in_parallel(data_alex_prepocessed, outputdata_path, endpoint_url, processes)
        time.sleep(1) # saving files takes some time..
        # Merging results into one file
        merge_and_save(outputnames, outputdata_path)
        # Delete intermediate files if needed
        if delete_intermediate:
            for filepath in outputnames:
                os.remove(filepath)
                print(f"Deleted {filepath}")
        print("Finished postprocessing triples for OpenAlex KG\n")
    else:
      raise Exception("Process number must be above 1 and integer")

def main():
  """
    To run this script direcly run:
        python -m src.data.data_extraction.triple_extraction.openalex.postprocess_dataset_alex   
    from the root directory of this project 
  """
  pre_processed_data_name = "data/interim/alex/pre_processed_data10.json"
  outputdata_name =  "data/interim/alex/post_processed_data10.json"
  endpoint_url = "https://semoa.skynet.coypu.org/sparql"#"https://semopenalex.org/sparql"

  post_process_alex_parallelized(pre_processed_data_name,outputdata_name,endpoint_url,8)

##############################################################################
if __name__ == "__main__":
    main()
