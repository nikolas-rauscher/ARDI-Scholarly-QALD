import json
import os

outputdata_name_path_0 = "data/processed/processed_data3_0.json"
outputdata_name_path_1 = "data/processed/processed_data3_1.json"

def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

data_0 = read_json_(outputdata_name_path_0)
data_1 = read_json_(outputdata_name_path_1)

data = data_0 + data_1 


take_label= ["authored by", "edited by", "differentFrom"]
take_object=["affiliation","primary affiliation", "creator note"]
link= ["wikidata","homepage URL","primary homepage URL","web page URL", "wikipedia page URL", "archived page URL", "award page URL"]
predicate= ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]
skip_labels= ["signature creator", "sameAs", "full creator name", "orcid", "primary full creator name", "signature creator"]
skip_uris = ["http://www.w3.org/2000/01/rdf-schema#label", "http://purl.org/spar/datacite/hasIdentifier"]

post_processed_data= []

for question in data:
    questions_dict={}
    questions_dict["id"] = question["id"]
    questions_dict["question"] = question["question"]
    questions_dict["answer"] = question["answer"]
    questions_dict["author_dblp_uri"] = question["author_dblp_uri"]

    new_tripples= []
    for tripple in question["all_tripples"]:
        new_tripple_dict = {}
        if tripple["predicate"] in predicate:
            new_tripple_dict["subject"] = tripple["subjectLabel"]
            new_tripple_dict["predicate"] = "is"
            new_tripple_dict["object"] = tripple["objectLabel"]
        elif tripple["predicateLabel"] in take_label:
            new_tripple_dict["subject"] = tripple["subjectLabel"]
            new_tripple_dict["predicate"] = tripple["predicateLabel"]
            new_tripple_dict["object"] = tripple["objectLabel"]
        elif tripple["predicateLabel"] in take_object:
            new_tripple_dict["subject"] = tripple["subjectLabel"]
            new_tripple_dict["predicate"] = tripple["predicateLabel"]
            new_tripple_dict["object"] = tripple["object"]
        elif tripple["predicateLabel"] in link:
            new_tripple_dict["subject"] = tripple["subjectLabel"]
            new_tripple_dict["predicate"] = "link"
            new_tripple_dict["object"] = tripple["object"]
        elif tripple["predicateLabel"] in skip_labels or tripple["predicate"] in skip_uris :
            pass 
        else:
            print(f"undefined id: {tripple['predicate']} for questionid: {question['id']}")
        if new_tripple_dict: new_tripples.append(new_tripple_dict)

    questions_dict["tripples_number"] = len(new_tripples)
    questions_dict["all_tripples"] = new_tripples

    post_processed_data.append(questions_dict)

# Save the new processed training data
save_processed_data_path = "data"
outputdata_name = "processed_data.json"
file_path = os.path.join(save_processed_data_path, outputdata_name)
with open(file_path, 'w') as file:
    json.dump(post_processed_data, file, indent=4,ensure_ascii=False)



                 

             
            