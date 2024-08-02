from main import create_dataset
import json


question = "What is the field of study of Tom Wilson?"
author_dblp_uri = ["https://dblp.org/pid/w/TDWilson"]
experiment_name = "test"

RAW_Wikipedia_path = "data/external/wiki_data.txt"

DBLP_endpoint_url ="https://dblp-april24.skynet.coypu.org/sparql" #SPARQL endpoint URL
DBLP_name_outputdata_PREprocessed = "pre_processed" + experiment_name + ".json"
DBLP_name_outputdata_POSTprocessed ="post_processed" + experiment_name + ".json"

OpenAlex_endpoint_url ="https://semopenalex.org/sparql" # Alternative smaller KG: "https://semoa.skynet.coypu.org/sparql"
OpenAlex_name_outputdata_PREprocessed = "pre_processed" + experiment_name + ".json"
OpenAlex_name_outputdata_POSTprocessed = "post_processed" + experiment_name + ".json"

Wikipedia_name_outputdata = "wiki_data_"+ experiment_name + ".json"

Final_name_merged_data = "final_merged_data" + experiment_name+ ".json"


def create_question_dict(question,author_dblp_uri,experiment_name):

    question={      
        "id": 1,
        "question": question,
        "answer": "-",
        "author_dblp_uri": str(author_dblp_uri) 
     }
    
    save_path = "data/raw/" + experiment_name
    with open(save_path, 'w') as file:
        json.dump(question, file, indent=4,ensure_ascii=False)

    create_dataset()