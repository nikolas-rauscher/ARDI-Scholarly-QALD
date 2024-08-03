from .helper_functions import ngram_cosine_similarity, read_json, save_intermediate_result
from .process_wikidata import prepare_wikipedia_data
import os


def extract_entity_from_wiki_text(text, substring: int):
    """
    Extracts a sequence of words starting from the beginning of the provided text.
    The sequence includes all initial words up to a specified count and continues to include
    subsequent words only if they start with an uppercase letter.

    Parameters:
        - text (str): The text from which to extract the entity.
        - substring (int): The number of initial words to include before checking for uppercase initials.

    Returns:
        - str: A string of the extracted entity or a message indicating insufficient words.

    Example:
        >>> extract_entity_from_wiki_text("John Doe Van Lufer is a scientists", 3)
    'John Doe Van Lufer'
    """

    words = text.split()    
    # Start from the first three words
    if len(words) < substring:
        return "Not enough words"
    
    result = words[:substring] 
    for word in words[3:]:
        # Check if the first letter is uppercase
        if word[0].isupper():
            result.append(word)
        else:
            break  # Stop appending if a word starts with a lowercase letter
    return ' '.join(result)



def find_wiki_article_by_name(names: list, data_dict: list) -> dict:
    """
    Searches through a list of Wikipedia articles to find the best match for each name in the provided list based on cosine similarity of n-grams.

    Parameters:
        - names (list): A list of names (strings) to be matched against Wikipedia article texts.
        - data_dict (list): A list of dictionaries containing Wikipedia articles, where each dictionary has text data.

    Returns:
        - dict: The dictionary of the most matching Wikipedia article based on the highest accumulated match score across all provided names.
             Returns an empty dictionary if no matches exceed the similarity threshold.
    """
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
   


def find_wiki_article_by_institution(institutions: list, data_dict: list) -> list:
    """
    Searches through a list of Wikipedia articles to find the ones that best match each institution in the provided list based on cosine similarity of n-grams.

    Parameters:
        - institutions (list): A list of institution names (strings) to be matched against Wikipedia article texts.
        - data_dict (list): A list of dictionaries containing Wikipedia articles, where each dictionary has text data.

    Returns:
        - list: A list of dictionaries for the most matching Wikipedia articles based on the highest cosine similarity score that exceeds a threshold of 0.5.
    """
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
    return found_articles


    

def add_wikidata(raw_wiki_data_path: str, processed_wiki_data_path: str, alex_data_path: str,  outputdata_path: str):
    """
    Integrates Wikipedia data into an Alex dataset by matching entities based on names and institutional affiliations.

    Parameters:
        - raw_wiki_data_path (str): Path to the raw Wikipedia data file.
        - processed_wiki_data_path (str): Path to the processed Wikipedia data file.
        - alex_data_path (str): Path of the Alex dataset to be enhanced with Wikipedia data.
        - outputdata_path (str): Path for the output dataset with added Wikipedia data.

    Description:
        This function first checks if the processed Wikipedia data exists. If not, it processes the raw Wikipedia data.
        It then reads the processed Wikipedia and Alex datasets, and iterates over each question in the Alex dataset.
        For each entity in the dataset, it extracts names and institutions and searches for matching Wikipedia articles.
        It adds these articles to the 'wiki_data' field in each question and saves the enhanced dataset to a specified location.
    """

    print("Extracting wikipedia articles..\n")
    if not os.path.exists(processed_wiki_data_path):
        prepare_wikipedia_data(raw_wiki_data_path, processed_wiki_data_path)
    wiki_data = read_json(processed_wiki_data_path)
    alex_data = read_json(alex_data_path)
    new_dataset = []
    for question, i in zip(alex_data, range(len(alex_data))):
        print(i,"/",len(alex_data))
        question_dict = {}
        question_dict["question"] = question["question"]
        question_dict["id"] = question["id"]
        question_dict["author_uri"] = question["author_uri"]
        wiki_articles= []
        for entity in question["all_triples"]:
            names = []
            institutions = []
            for triple in entity["triples"]:
                if triple["predicate"] == "alternativeName":
                    names.append(triple["object"])
                if triple["predicate"] == "was working in":
                    institution = triple["object"].split(" while writing paper")[0]
                    institutions.append(institution)
                    institutions = list(set(institutions))
            
            wiki_articles.append(find_wiki_article_by_name(names, wiki_data))
            wiki_articles = wiki_articles + find_wiki_article_by_institution(institutions, wiki_data)

            question_dict["wiki_data"]= wiki_articles
        
        new_dataset.append(question_dict)
        save_intermediate_result(outputdata_path, new_dataset)
    print("Finished extracting wikipedia articles..\n")



def main():
    """
    To run this script direcly run:
        python -m src.data.data_extraction.wikipedia_data.add_wiki_data
    from the root directory of this project 
    """
    alex_data_name = "data/interim/alex/post_processed_data10.json"
    raw_wiki_data_path = "data/external/wiki_data.txt"
    processed_wiki_data_path = "data/interim/wikipedia_data/processed_wikipedia_data.txt"
    outputdata_name =  "data/interim/wikipedia_data/wikipedia_data_extracted.json"

    add_wikidata(raw_wiki_data_path,processed_wiki_data_path, alex_data_name, outputdata_name)
##############################################################################
if __name__ == "__main__":
    main()


