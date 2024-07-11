import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_json_(outputdata_name_path) -> dict:
    with open(outputdata_name_path, 'r', encoding='utf-8') as file:
        formulations = json.load(file)
    return formulations

def save_intermediate_result(outputdata_name, new_dataset):
    with open(outputdata_name, 'w') as file:
        json.dump(new_dataset, file, indent=4, ensure_ascii=False)



def extract_entity_from_wiki_text(text, substring: int):
    # Split the text into words
    words = text.split()
    
    # Start from the first three words
    if len(words) < substring:
        return "Not enough words"
    
    result = words[:substring]  # Capture the first three words initially
    
    # Start processing from the fourth word (index 3)
    for word in words[3:]:
        # Check if the first letter is uppercase
        if word[0].isupper():
            result.append(word)
        else:
            break  # Stop appending if a word starts with a lowercase letter

    # Join the result list back into a string
    return ' '.join(result)



def find_wiki_article_by_name(names:list, data_dict) -> dict:
    scores = []
    match_table={}
    for name in names:
        for article, id_number in zip(data_dict,range(len(data_dict))):
            article=article[0].values()
            article = list(article)[0]  # Convert the values view to a list
            entity = extract_entity_from_wiki_text(article,3) 

            score=ngram_cosine_similarity(name, entity, n=3)
            if score >0.3:
                if id_number not in match_table:
                    match_table[id_number]= 1
                else:
                    match_table[id_number]= match_table[id_number] +1
                scores.append(score)
    if match_table:          
        match = max(match_table, key=match_table.get)
        return data_dict[match][0]
    else:
        return []
   


def find_wiki_article_by_institution(institutions:list, data_dict) -> list:
    match_table={}
    found_articles = []
    all_matches=[]
    for institution in institutions:
        for article, id_number in zip(data_dict,range(len(data_dict))):
            article=article[0].values()
            article = list(article)[0]  # Convert the values view to a list
            entity = extract_entity_from_wiki_text(article,10) 

            score=ngram_cosine_similarity(institution, entity, n=3)
            if score >0.5:
                match_table[id_number]= score
              
        if match_table:
            match = max(match_table, key=match_table.get)
            all_matches.append(match)
    final_matches = list(set(all_matches))
    for match_id in final_matches: found_articles.append(data_dict[match_id][0])
    #print(found_articles)
    return found_articles


    


def ngram_cosine_similarity(s1, s2, n=3):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([s1, s2])
    return cosine_similarity(ngrams)[0, 1]



alex_data = read_json_("data/processed/alex/post_processed_data500.json")
wiki_data = read_json_("data/external/wiki_data_processed.txt")
#alex_data = alex_data[:1]  # Limiting to the first item for processing as per the example

new_dataset = []

for question, i in zip(alex_data, range(len(alex_data))):
    print(i)
    wiki_articles= []
    for entity in question["all_tripples"]:
        names = []
        institutions = []
        for triple in entity["tripples"]:
            if triple["predicate"] == "alternativeName":
                names.append(triple["object"])
            if triple["predicate"] == "was working in":
                institution = triple["object"].split(" while writing paper")[0]
                institutions.append(institution)
                institutions = list(set(institutions))
        
        wiki_articles.append(find_wiki_article_by_name(names, wiki_data))
        wiki_articles = wiki_articles + find_wiki_article_by_institution(institutions, wiki_data)
        question["wiki_data"]= wiki_articles
    
    new_dataset.append(question)
    #print(f'Processed question id: {question["id"]}')

    save_intermediate_result("data/processed/final/post_processed_data_alex+wiki500.json", new_dataset)