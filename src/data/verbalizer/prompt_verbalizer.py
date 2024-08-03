import json
from tqdm import tqdm
from collections import defaultdict
from .verbalisation_module import VerbModule

def group_triples(tripleList):
    """
    Groups triples by their predicates, collecting subjects and objects into sets.

    Args:
        tripleList (list): A list of triples, where each triple is a dictionary with keys 'subject', 'predicate', and 'object'.

    Returns:
        dict: A dictionary where keys are predicates and values are dictionaries with 'subjects' and 'objects' keys containing sets of subjects and objects, respectively.
    """
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
    """
    Formats the triples in the given predicate dictionary.

    Args:
        predicateDict (dict): A dictionary containing predicates as keys and entities as values.

    Returns:
        list: A list of formatted triples.
    """
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
    """
    Preprocesses a list of triples by sorting them, grouping by predicates, and formatting them.

    Args:
        tripleList (list): A list of triples, where each triple is a dictionary with keys 'subject', 'predicate', and 'object'.

    Returns:
        tuple: A tuple containing a list of formatted triples and the predicate dictionary.
    """
    tripleList.sort(key=lambda x: x['predicate'])
    predicateDict = group_triples(tripleList)
    formatted_triples = format_triples(predicateDict)
    return formatted_triples, predicateDict

def verbalise_by_predicate(predicateDict, verbModule):
    """
    Verbalises triples by predicate using a given verbalisation module.

    Args:
        predicateDict (dict): A dictionary containing predicates as keys and entities as values.
        verbModule (VerbModule): An instance of the VerbModule class for verbalising the triples.

    Returns:
        list: A list of verbalised triples as strings.
    """
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
    """
    Verbalises a list of triples.

    Args:
        triples (list): A list of triples, where each triple is a dictionary with keys 'subject', 'predicate', and 'object'.
        use_predicate_based (bool): Whether to use predicate-based verbalisation.

    Returns:
        str: A string of verbalised triples.
    """
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
