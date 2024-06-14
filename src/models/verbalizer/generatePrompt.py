import json
from tqdm import tqdm
from verbalisation_module import VerbModule

def verbalise(tripleList, verbModule):
    ans = "translate Graph to English: "
    predicateSet = set([])
    for item in tripleList:
        if item['predicate'] not in predicateSet or item['predicate'] == "instance of":
            
            if "object" in item.keys():
                ans += f"<H> {item['subject']} <R> {item['predicate']} <T> {item['object']} "
            elif "prob" in item.keys():
                ans += f"<H> {item['subject']} <R> {item['predicate']} <T> {item['prob']} "
            predicateSet.add(item['predicate'])
    return verbModule.verbalise(ans)

def plainPrompt(tripleList):
    ans = ""
    predicateSet = set([])
    for item in tripleList:
        if item['predicate'] not in predicateSet or item['predicate'] == "instance of":
            if "object" in item.keys():
                ans += f"{item['subject']} {item['predicate']} {item['object']}. "
            elif "prob" in item.keys():
                ans += f"{item['subject']} {item['predicate']} {item['prob']}. "
            predicateSet.add(item['predicate'])
    return ans

def verbaliseFile(FILENAME, outputFile, limit):
    results = []
    f = open(FILENAME, "r")
    data = json.loads(f.read())
    f.close() 
    verb_module = VerbModule() 
    for item in tqdm(data[:limit]):  
        oneItem = {}
        oneItem['id'] = item['id'] 
        oneItem['question'] = item['question']
        oneItem['answer'] = item['answer']
        oneItem['author_dblp_uri'] = item['author_dblp_uri']
        oneItem['plain_prompt'] = plainPrompt(item['all_tripples']) 
        oneItem['verbalised_prompt'] = verbalise(item['all_tripples'], verb_module)
        results.append(oneItem)
    with open(outputFile, "w", encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    FILENAME = "exp10.json"
    FILENAME = "processed_data.json"
    outputFile = "exp10_out.json"
    verbaliseFile(FILENAME, outputFile, limit=3) # add limit for testing