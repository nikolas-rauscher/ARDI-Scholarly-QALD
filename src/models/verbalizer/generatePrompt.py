import json
from tqdm import tqdm
from collections import defaultdict
from verbalisation_module import VerbModule

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
    return formatted_triples

def verbalise(tripleList, verbModule):
    final_ans_list = []
    predicateDict = group_triples(tripleList)
    formatted_triples = format_triples(predicateDict)

    for predicate, entities in predicateDict.items():
        ans = "translate Graph to English: "
        subjects = list(entities['subjects'])
        objects = list(entities['objects'])
        
        if len(subjects) == 1 and len(objects) > 5:
            ans += f'<H> {subjects[0]} <R> "{predicate}" <T> "{", ".join(objects)}"'
        elif len(objects) == 1 and len(subjects) > 5:
            ans += f'<H> "{", ".join(subjects)}" <R> "{predicate}" <T> {objects[0]}'
        else:
            for subj in subjects:
                for obj in objects:
                    ans += f'<H> {subj} <R> "{predicate}" <T> {obj}'
        
        final_ans_list.append(verbModule.verbalise(ans))

    return final_ans_list

def plainPrompt(tripleList):
    ans = ""
    predicateDict = group_triples(tripleList)
    formatted_triples = format_triples(predicateDict)

    for item in formatted_triples:
        if isinstance(item['object'], list):
            ans += f'{item["subject"]} "{item["predicate"]}" "{", ".join(item["object"])}".'
        elif isinstance(item['subject'], list):
            ans += f'"{", ".join(item["subject"])}" "{item["predicate"]}" {item["object"]}.'
        else:
            ans += f'{item["subject"]} "{item["predicate"]}" {item["object"]}.'
        ans += "\n"
    return ans.strip()

def verbaliseFile(FILENAME, outputFile, limit):
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

        preprocessed_triples = preprocess_triples(item['all_tripples'])

        oneItem['plain_prompt'] = plainPrompt(preprocessed_triples) 
        verbalised_list = verbalise(preprocessed_triples, verb_module)
        oneItem['verbalised_prompt'] = "\n".join(verbalised_list)

        oneItem['triples'] = {
            "preprocessed": preprocessed_triples,
            "verbalised": verbalised_list
        }
        results.append(oneItem)
        
    with open(outputFile, "w", encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    FILENAME = "processed_data.json"
    outputFile = "verbalised_data.json"
    verbaliseFile(FILENAME, outputFile, limit=1) # add limit for testing