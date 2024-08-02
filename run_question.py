from main import create_dataset
import json


question = "What is the field of study of Tom Wilson?"
author_dblp_uri = ["https://dblp.org/pid/w/TDWilson"]
experiment_name = "test"



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