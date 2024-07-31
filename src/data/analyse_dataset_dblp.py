import json
import os


def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations



#get all predicates labels
def get_all_predicate_labels(data):
    predicates_labels= []
    predicates = []
    for question,i in zip(data, range(len(data))):
        print(i)
        for item in question["all_triples"]:
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
        for item in question["all_triples"]:
            if item["predicate"] not in predicates and item["predicateLabel"] == "" :
                predicates.append(item["predicate"])
                predicates_labels.append(item["predicateLabel"])
    print(predicates)
    print(len(predicates))
    return predicates


# get all authors without oridID

def get_authors_without_oridID(data):
    author_list = []
    for question in data:
        for item in question["all_triples"]:
            if item["objectLabel"] == "Person":
                author_list.append(item["subject"])
    print (len(author_list))

    for question in data:
        for item in question["all_triples"]:
            if item["predicateLabel"] == "orcid":
                subject = item["subject"]
                if subject in author_list:
                    author_list.remove(subject)   
    print(len(author_list)) 
    return      
        

def main():

    outputdata_name_path_0 = "/Users/erikrubinov/Desktop/SM24/project/repos/processed_data3_0.json"
    outputdata_name_path_1 = "/Users/erikrubinov/Desktop/SM24/project/repos/processed_data3_1.json"
    data_0 = read_json_(outputdata_name_path_0)
    data_1 = read_json_(outputdata_name_path_1)
    data = data_0 + data_1

  

    #get_all_predicate_labels(data)
    #get_all_predicates_without_labels(data)
    get_authors_without_oridID(data)
 

##############################################################################
if __name__ == "__main__":
    main()

    


