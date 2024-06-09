import json
import os


def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations



#get all predicates
def get_all_predicate_labels(data):
    predicates_labels= []
    predicates = []
    for question,i in zip(data, range(len(data))):
        print(i)
        for item in question["all_tripples"]:
            if item["predicate"] not in predicates:
                predicates.append(item["predicate"])
                predicates_labels.append(item["predicateLabel"])
    print(predicates_labels)
    print(len(predicates_labels))
    return predicates_labels


#get all predicates
def get_all_predicates_without_labels(data):
    predicates_labels= []
    predicates = []
    for question,i in zip(data, range(len(data))):
        print(i)
        for item in question["all_tripples"]:
            if item["predicate"] not in predicates and item["predicateLabel"] == "" :
                predicates.append(item["predicate"])
                predicates_labels.append(item["predicateLabel"])
    print(predicates)
    print(len(predicates))
    return predicates


def main():

    outputdata_name_path_0 = "data/processed/processed_data3_0.json"
    outputdata_name_path_1 = "data/processed/processed_data3_1.json"
    data_0 = read_json_(outputdata_name_path_0)
    data_1 = read_json_(outputdata_name_path_1)
    data = data_0 + data_1
    #get_all_predicate_labels(data)
    get_all_predicates_without_labels(data)
 

##############################################################################
if __name__ == "__main__":
    main()

    


