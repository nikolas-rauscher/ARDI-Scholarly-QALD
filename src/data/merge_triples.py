import json
def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

data1 = read_json_("data/processed/alex/post_processed_data0-99.json")
data2 = read_json_("data/processed/alex/post_processed_data100-200.json")
data3 = read_json_("data/processed/alex/post_processed_data201-300.json")
data4 = read_json_("data/processed/alex/post_processed_data301-400.json")
data5 = read_json_("data/processed/alex/post_processed_data401-500.json")
alex_parts = [data1,data2,data3,data4,data5]



merged_data=alex_parts[0]
for part in alex_parts[1:]:
   merged_data = merged_data+part

with open("data/processed/alex/post_processed_data500.json", 'w') as file: 
        json.dump(merged_data, file, indent=4, ensure_ascii=False)


#read dblp data
dblp_data = read_json_("data/processed/dblp/post_processed_data1000.json") 
alex_data = read_json_("data/processed/alex/post_processed_data500.json")
dblp_data = dblp_data[:500]
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

    new_merged_question["triples_number"]  = dblp_question["tripples_number"] + alex_question["tripples_number"]

    for entity_alex,entity_dblp in zip(alex_question["all_tripples"],dblp_question["all_tripples"]):
        merged_triples_all = []
        merged_triples_for_entity = entity_alex["tripples"] + entity_dblp["tripples"]
        merged_triples_all.append(merged_triples_for_entity)
    

    new_merged_question["all_triples"] = merged_triples_all
    new_merged_dataset.append(new_merged_question)


with open("data/processed/final/processed_data500.json", 'w') as file: 
        json.dump(new_merged_dataset, file, indent=4, ensure_ascii=False)


        
           

