import json
from tqdm import tqdm
import sys
sys.path.append('./src')
from collections import defaultdict
from models.verbalizer.verbalisation_module import VerbModule
import os

def group_triples(tripleList):
    predicateDict = defaultdict(lambda: defaultdict(set))
    for item in tripleList:
        subjects = item['subject'] if isinstance(item['subject'], list) else [item['subject']]
        objects = item['object'] if isinstance(item['object'], list) else [item['object']]
        for subj in subjects:
            predicateDict[item['predicate']]['subjects'].add(subj)
        for obj in objects:
            predicateDict[item['predicate']]['objects'].add(obj)
    return predicateDict

def format_triples(predicateDict):
    formatted_triples = []
    large_lists = []

    for predicate, entities in sorted(predicateDict.items()):
        subjects = list(entities['subjects'])
        objects = list(entities['objects'])

        if len(subjects) == 1 and len(objects) > 5:
            large_lists.append({"subject": subjects[0], "predicate": predicate, "object": objects})
        elif len(objects) == 1 and len(subjects) > 5:
            large_lists.append({"subject": subjects, "predicate": predicate, "object": objects[0]})
        else:
            for subj in subjects:
                for obj in objects:
                    formatted_triples.append({"subject": subj, "predicate": predicate, "object": obj})

    formatted_triples.sort(key=lambda x: x['predicate'])  # Sort the normal triples by predicate
    large_lists.sort(key=lambda x: x['predicate'])  # Sort the large lists by predicate

    formatted_triples.extend(large_lists)  # Add the large lists at the end
    return formatted_triples

def preprocess_triples(tripleList):
    tripleList.sort(key=lambda x: x['predicate'])
    predicateDict = group_triples(tripleList)
    formatted_triples = format_triples(predicateDict)
    return formatted_triples, predicateDict

def verbalise_by_predicate(predicateDict, verbModule):
    final_ans_list = []

    for predicate, entities in predicateDict.items():
        ans = "translate Graph to English: "
        subjects = list(entities['subjects'])
        objects = list(entities['objects'])
        
        if len(subjects) == 1 and len(objects) > 5:
            ans += f'<H> {subjects[0]} <R> {predicate} <T> "{", ".join(objects)}"'
        elif len(objects) == 1 and len(subjects) > 5:
            ans += f'<H> "{", ".join(subjects)}" <R> {predicate} <T> {objects[0]}'
        else:
            for subj in subjects:
                for obj in objects:
                    ans += f'<H> {subj} <R> {predicate} <T> {obj}'
        
        # Call the verbModule's verbalise method for the constructed string
        final_ans_list.append(verbModule.verbalise(ans))

    return final_ans_list

def plainPrompt(formatted_triples):
    ans = ""

    for item in formatted_triples:
        if isinstance(item['object'], list):
            ans += f'{item["subject"]} {item["predicate"]} "{", ".join(item["object"])}".'
        elif isinstance(item['subject'], list):
            ans += f'"{", ".join(item["subject"])}" {item["predicate"]} {item["object"]}.'
        else:
            ans += f'{item["subject"]} {item["predicate"]} {item["object"]}.'
        ans += "\n"
    return ans.strip()

def verbalise_all_at_once(tripleList, verbModule):
    ans = "translate Graph to English: "
    for item in tripleList:
        ans += f'<H> {item["subject"]} <R> {item["predicate"]} <T> {item["object"]}'
    return [verbModule.verbalise(ans)]

def verbaliseFile(FILENAME, outputFile, limit, use_predicate_based_verbalisation=True, include_preprocessed=False):
    results = []
    with open(FILENAME, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    verb_module = VerbModule()
    for item in tqdm(data[:limit]):  
        oneItem = {}
        oneItem['id'] = item['id']
        oneItem['question'] = item['question']
        oneItem['answer'] = item['answer']
        oneItem['author_dblp_uri'] = item['author_dblp_uri']

        # Preprocess the triples (sort and format)
        preprocessed_triples, predicate_dict = preprocess_triples(item['all_tripples'])

        # Generate plain prompt
        oneItem['plain_prompt'] = plainPrompt(preprocessed_triples) 
        
        # Generate verbalised prompt based on the flag
        if use_predicate_based_verbalisation:
            verbalised_list = verbalise_by_predicate(predicate_dict, verb_module)
            oneItem['verbalised_prompt'] = "\n".join(verbalised_list)
        else:
            verbalised_list = verbalise_all_at_once(preprocessed_triples, verb_module)
            oneItem['verbalised_prompt'] = "\n".join(verbalised_list)

        # Optionally include preprocessed triples in the output
        if include_preprocessed:
            oneItem['preprocessed_triples'] = preprocessed_triples

        results.append(oneItem)
        
    with open(outputFile, "w", encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

# if __name__ == "__main__":
#     FILENAME = "processed_data.json"
#     outputFile = "verbalised_data.json"
#     verbaliseFile(FILENAME, outputFile, limit=1, use_predicate_based_verbalisation=True, include_preprocessed=True)  # add limit for testing

def verbalise_triples(triples, use_predicate_based_verbalisation=True):
    verb_module = VerbModule()
    
    # Preprocess the triples (sort and format)
    preprocessed_triples, predicate_dict = preprocess_triples(triples)

    # Generate plain prompt
    plain_prompt = plainPrompt(preprocessed_triples)
    
    # Generate verbalised prompt based on the flag
    if use_predicate_based_verbalisation:
        verbalised_list = verbalise_by_predicate(predicate_dict, verb_module)
        verbalised_prompt = "\n".join(verbalised_list)
    else:
        verbalised_list = verbalise_all_at_once(preprocessed_triples, verb_module)
        verbalised_prompt = "\n".join(verbalised_list)

    # return {
    #     'plain_prompt': plain_prompt,
    #     'verbalised_prompt': verbalised_prompt,
    #     'preprocessed_triples': preprocessed_triples 
    # }
    return verbalised_prompt

# Test
if __name__ == "__main__":

    triples = [
{"subject": "John", "predicate": "likes", "object": "apples"},
        {"subject": "Mary", "predicate": "likes", "object": "oranges"},
        {"subject": "Mary", "predicate": "likes", "object": "bananas"},
        {"subject": "Alice", "predicate": "knows", "object": "Charlie"},
        {"subject": "Bob", "predicate": "knows", "object": "Charlie"},
        {"subject": "Bob", "predicate": "knows", "object": "Dave"},
        {"subject": "Bob", "predicate": "knows", "object": "Eve"},
        {"subject": "Charlie", "predicate": "works_with", "object": "Eve"},
        {"subject": "Dave", "predicate": "works_with", "object": "Eve"}
    ]
    
    result = verbalise_triples(triples, use_predicate_based_verbalisation=True)
    print(result)