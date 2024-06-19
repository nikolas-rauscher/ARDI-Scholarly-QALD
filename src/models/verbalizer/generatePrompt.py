import json
from tqdm import tqdm
from collections import defaultdict
from verbalisation_module import VerbModule

def group_triples(tripleList):
    predicateDict = defaultdict(lambda: defaultdict(set))
    for item in tripleList:
        predicateDict[item['predicate']]['subjects'].add(item['subject'])
        predicateDict[item['predicate']]['objects'].add(item['object'])
    return predicateDict

def format_triples(predicateDict):
    formatted_triples = []
    large_lists = []

    for predicate, entities in sorted(predicateDict.items()):
        subjects = list(entities['subjects'])
        objects = list(entities['objects'])

        if len(subjects) == 1 and len(objects) > 5:
            large_lists.append({"subject": subjects[0], "predicate": predicate, "object": list(objects)})
        elif len(objects) == 1 and len(subjects) > 5:
            large_lists.append({"subject": list(subjects), "predicate": predicate, "object": objects[0]})
        else:
            for subj in subjects:
                for obj in objects:
                    formatted_triples.append({"subject": subj, "predicate": predicate, "object": obj})

    formatted_triples.sort(key=lambda x: x['predicate'])  # Sort the normal triples by predicate
    large_lists.sort(key=lambda x: x['predicate'])  # Sort the large lists by predicate

    formatted_triples.extend(large_lists)  # Add the large lists at the end
    return formatted_triples

def print_triples(triples, title):
    print(title)
    for triple in triples:
        print(triple)
        print("\n")

def preprocess_triples(tripleList):
    # Sort triples by predicate
    tripleList.sort(key=lambda x: x['predicate'])
    
    predicateDict = group_triples(tripleList)
    formatted_triples = format_triples(predicateDict)
    
    # Print the sorted form
    print_triples(tripleList, "Sorted Triples:")
    
    # Print the grouped and formatted triples
    print_triples(formatted_triples, "Grouped Triples:")
    
    return formatted_triples

def verbalise(tripleList, verbModule):
    ans = "translate Graph to English: "
    for item in tripleList:
        if isinstance(item['object'], list):
            ans += f"<H> {item['subject']} <R> {item['predicate']} <T> {', '.join(item['object'])} "
        elif isinstance(item['subject'], list):
            ans += f"<H> {', '.join(item['subject'])} <R> {item['predicate']} <T> {item['object']} "
        else:
            ans += f"<H> {item['subject']} <R> {item['predicate']} <T> {item['object']} "
        ans += "\n"
    return verbModule.verbalise(ans)

def plainPrompt(tripleList):
    ans = ""
    for item in tripleList:
        if isinstance(item['object'], list):
            ans += f"{item['subject']} {item['predicate']} {', '.join(item['object'])}. "
        elif isinstance(item['subject'], list):
            ans += f"{', '.join(item['subject'])} {item['predicate']} {item['object']}. "
        else:
            ans += f"{item['subject']} {item['predicate']} {item['object']}. "
        ans += "\n"
    return ans

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
        oneItem['verbalised_prompt'] = verbalise(preprocessed_triples, verb_module)
        results.append(oneItem)
    with open(outputFile, "w", encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    FILENAME = "processed_data.json"
    outputFile = "verbalised_data.json"
    verbaliseFile(FILENAME, outputFile, limit=3) # add limit for testing