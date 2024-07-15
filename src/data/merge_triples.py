import json
def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

data1 = read_json_("data/processed/alex/pre_processed_data1000_0-99.json")
data1_ = read_json_("data/processed/alex/pre_processed_data1000_0-99-2.json")
data1__ = read_json_("data/processed/alex/pre_processed_data1000_0-99-3.json")

data2 = read_json_("data/processed/alex/pre_processed_data1000_100-199.json")
data2_ = read_json_("data/processed/alex/pre_processed_data1000_100-199-2.json")
data2__ = read_json_("data/processed/alex/pre_processed_data1000_100-199-3.json")

data3 = read_json_("data/processed/alex/pre_processed_data1000_200-299.json")
data3_ = read_json_("data/processed/alex/pre_processed_data1000_200-299-2.json")
data3__ = read_json_("data/processed/alex/pre_processed_data1000_200-299-3.json")

data4 = read_json_("data/processed/alex/pre_processed_data1000_300-399.json")
data4_ = read_json_("data/processed/alex/pre_processed_data1000_300-399-2.json")
data4__ = read_json_("data/processed/alex/pre_processed_data1000_300-399-3.json")

data5 = read_json_("data/processed/alex/pre_processed_data1000_400-500.json")
data5_ = read_json_("data/processed/alex/pre_processed_data1000_400-499-2.json")
data5__ = read_json_("data/processed/alex/pre_processed_data1000_400-499-3.json")

data6 = read_json_("data/processed/alex/pre_processed_data1000_500-599.json")
data6_ = read_json_("data/processed/alex/pre_processed_data1000_500-599-2.json")
data6__ = read_json_("data/processed/alex/pre_processed_data1000_500-599-3.json")

data7 = read_json_("data/processed/alex/pre_processed_data1000_600-699.json")
data7_ = read_json_("data/processed/alex/pre_processed_data1000_600-699-2.json")
data7__ = read_json_("data/processed/alex/pre_processed_data1000_600-699-3.json")

data8 = read_json_("data/processed/alex/pre_processed_data1000_700-799.json")
data8_ = read_json_("data/processed/alex/pre_processed_data1000_700-799-2.json")
data8__ = read_json_("data/processed/alex/pre_processed_data1000_700-799-3.json")

data9 = read_json_("data/processed/alex/pre_processed_data1000_800-899.json")
data9_ = read_json_("data/processed/alex/pre_processed_data1000_800-899-2.json")
data9__ = read_json_("data/processed/alex/pre_processed_data1000_800-899-3.json")

data10 = read_json_("data/processed/alex/pre_processed_data1000_900-1000.json")
data10_ = read_json_("data/processed/alex/pre_processed_data1000_900-1000-2.json")
data10__ = read_json_("data/processed/alex/pre_processed_data1000_900-1000-3.json")

alex_parts = [data1, data1_, data1__, data2, data2_, data2__, data3, data3_, data3__, data4, data4_, data4__, data5, data5_, data5__, data6, data6_, data6__, data7, data7_, data7__, data8, data8_, data8__, data9, data9_, data9__, data10, data10_, data10__]

merged_data=alex_parts[0]
for part in alex_parts[1:]:
   merged_data = merged_data+part

with open("data/processed/alex/post_processed_data_part.json", 'w') as file: 
        json.dump(merged_data, file, indent=4, ensure_ascii=False)


#read dblp data
dblp_data = read_json_("data/processed/dblp/post_processed_data1000.json") 
alex_data = read_json_("data/processed/final/post_processed_data_alex+wiki500.json")
dblp_data = dblp_data[:500]
new_merged_dataset = []
for alex_question in alex_data:
    new_merged_question = {}
    new_merged_question["id"] = alex_question["id"]
    new_merged_question["question"] = alex_question["question"]
    new_merged_question["answer"] = alex_question["answer"]
    new_merged_question["wiki_data"] = alex_question["wiki_data"]

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


with open("data/processed/final/processed_data_final500.json", 'w') as file: 
        json.dump(new_merged_dataset, file, indent=4, ensure_ascii=False)


        
           

