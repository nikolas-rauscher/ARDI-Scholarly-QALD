import json

def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations




#get all predicates labels
def get_all_predicate_labels(data):
    predicates = []
    for question,i in zip(data, range(len(data))):
        print(i)
        for item in question["all_tripples"]:
            if item["predicate"] not in predicates:
                predicates.append(item["predicate"])
              
    print(predicates)
    print(len(predicates))
    return predicates

# all question are correct...
def find_broken_questions(data):
    for question in data:
        if "author_dblp_uri" not in question:
            print("ERROR") 
        
        entities = question["author_dblp_uri"]
        if entities == "":
            print("ERROR")  

        if entities[0] != "<" and entities[0] != "[":
            print("ERROR")

        
        if entities[0] == "[":
            entities = eval(entities)
            if entities == []:
                print("ERROR")
    

#data_1 = read_json_("data/processed/processed_data_alex_01.json")
data_raw = read_json_("data/raw/trainingdata.json")
#res = get_all_predicate_labels(data_1)
find_broken_questions(data_raw)

