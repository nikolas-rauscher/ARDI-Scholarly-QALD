
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

    formatted_triples.sort(key=lambda x: x['predicate'])
    large_lists.sort(key=lambda x: x['predicate'])

    formatted_triples.extend(large_lists)
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

        final_ans_list.append(verbModule.verbalise(ans))

    return final_ans_list

def verbalise_triples(triples, use_predicate_based=True):
    verb_module = VerbModule()
    preprocessed_triples, predicate_dict = preprocess_triples(triples)
    
    if use_predicate_based:
        verbalised_list = verbalise_by_predicate(predicate_dict, verb_module)
        return "\n".join(verbalised_list)
    else:
        return verb_module.verbalise(
            "translate Graph to English: " + " ".join(
                [f'<H> {item["subject"]} <R> {item["predicate"]} <T> {item["object"]}' for item in preprocessed_triples]
            )
        )