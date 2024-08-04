import json

def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations


def merge_data(dblp_processed_path: str, alex_processed_path: str, extracted_wiki_data_path: str, merged_data_path: str): 
    """
    Merges data from DBLP, Alex, and Wikipedia datasets into a single dataset based on their IDs.

    Args:
        dblp_processed_name (str): Filename for the processed DBLP dataset.
        alex_processed_name (str): Filename for the processed Alex dataset.
        extracted_wiki_data_name (str): Filename for the processed Wikipedia dataset.
        merged_data_name (str): Output filename for the merged dataset.

    Description:
        This function reads three datasets, finds corresponding entries by ID, and merges them.
        It outputs the merged result into a specified file in the 'final' directory.
    """
    dblp_data = read_json_(dblp_processed_path) 
    alex_data = read_json_(alex_processed_path)
    wikipedia_data = read_json_(extracted_wiki_data_path)
    new_merged_dataset = []
    for alex_question in alex_data:
        new_merged_question = {}
        new_merged_question["id"] = alex_question["id"]
        new_merged_question["question"] = alex_question["question"]
        new_merged_question["answer"] = alex_question["answer"]

        # find question in alex dataset 
        for dblp_question in dblp_data:
            if dblp_question["id"] == alex_question["id"]:
                break
        
        # find question in extracted wiki articles 
        for wiki_article in wikipedia_data:
              if wiki_article["id"] == alex_question["id"]: 
                break
              
        new_merged_question["triples_number"]  = dblp_question["triples_number"] + alex_question["triples_number"]

        for entity_alex,entity_dblp in zip(alex_question["all_triples"],dblp_question["all_triples"]):
            merged_triples_all = []
            merged_triples_for_entity = entity_alex["triples"] + entity_dblp["triples"]
            merged_triples_all.append(merged_triples_for_entity)
        try: 
            new_merged_question["all_triples"] = merged_triples_all
            new_merged_question["wiki_data"] = wiki_article["wiki_data"]
            new_merged_dataset.append(new_merged_question)
        except KeyError as e:
            print(f"wikidata for {new_merged_question['id']} is missing ") 

    with open(merged_data_path, 'w') as file: 
            json.dump(new_merged_dataset, file, indent=4, ensure_ascii=False)
    return new_merged_dataset

def main():
  """
    To run this script direcly run:
        python -m src.data_extraction.merge_data  
    from the root directory of this project 
  """
  dblp_processed_name = "post_processed_data10.json"
  alex_processed_name =  "post_processed_data10.json"
  extracted_wiki_data_name = "wiki_data_processed.json"
  merged_data_name = "final10.json"

  merge_data(dblp_processed_name,alex_processed_name,extracted_wiki_data_name,merged_data_name)

##############################################################################
if __name__ == "__main__":
    main()

      

