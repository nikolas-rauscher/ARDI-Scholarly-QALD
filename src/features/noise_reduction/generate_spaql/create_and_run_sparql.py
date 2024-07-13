import json
import requests
import os
from dotenv import load_dotenv
import utils
import answer_extraction
from SPARQLWrapper import SPARQLWrapper, JSON
import datetime
from time import sleep
from tqdm import tqdm

def construct_prompt(question, sim_questions, examples, author_dblp_uri, with_schema=False):
    example = ''
    if with_schema:
        if examples != 1:
            for item in sim_questions[:examples]:
                sim_question = item['similar_question']
                sparql = utils.post_process_query(item['similar_question_sparql'])
                entities = ', '.join(item['entities'])
                example += (f"Question: {sim_question}\n"
                            f"Author ID: {entities}\n"
                            f"Sparql: {sparql}\n")
        else:
            item = sim_questions[0]
            sim_question = item['similar_question']
            sparql = utils.post_process_query(item['similar_question_sparql'])
            entities = ', '.join(item['entities'])
            example += (f"Question: {sim_question}\n"
                        f"Author ID: {entities}\n"
                        f"Sparql: {sparql}\n")
        
        return f"""
            Task: Generate SPARQL queries to query the DBLP knowledge graph based on the provided schema definition.

            Schema Definition:
            1. **Classes**:
            - Entity
            - Creator (Person, Group)
            - Publication (Book, Article, Inproceedings, Incollection, Editorship, Reference, Data, Informal, Withdrawn)
            - Signature (AuthorSignature, EditorSignature)
            - Stream (Conference, Journal, Series, Repository)
            - VersionRelation

            2. **Properties**:
            - identifier, wikidata, webpage, orcid, creatorName, primaryCreatorName, affiliation, primaryAffiliation
            - awardWebpage, homepage, primaryHomepage, creatorOf, authorOf, editorOf, coCreatorWith, coAuthorWith
            - coEditorWith, homonymousCreator, possibleActualCreator, proxyAmbiguousCreator, signatureCreator, signatureDblpName
            - signatureOrcid, signatureOrdinal, signaturePublication, doi, isbn, omid, title, bibtexType, createdBy
            - authoredBy, editedBy, numberOfCreators, hasSignature, documentPage, primaryDocumentPage, listedOnTocPage
            - publishedInStream, publishedIn, publishedInSeries, publishedInSeriesVolume, publishedInJournal, publishedInJournalVolume
            - publishedInJournalVolumeIssue, publishedInBook, publishedInBookChapter, pagination, yearOfEvent, yearOfPublication
            - monthOfPublication, publishedBy, publishersAddress, thesisAcceptedBySchool, publicationNote, publishedAsPartOf
            - hasVersion, isVersion, isVersionOf, versionConcept, versionInstance, versionUri, versionLabel, versionOrdinal
            - streamTitle, primaryStreamTitle, formerStreamTitle, issn, iso4, indexPage, relatedStream, superStream, subStream
            - predecessorStream, successorStream

            Instructions:
            - Use only these predicates:
            - https://dblp.org/rdf/schema#authoredBy
            - https://dblp.org/rdf/schema#doi
            - https://dblp.org/rdf/schema#title
            - https://dblp.org/rdf/schema#yearOfPublication
            - https://dblp.org/rdf/schema#affiliation
            - https://dblp.org/rdf/schema#primaryAffiliation
            - https://dblp.org/rdf/schema#publishedIn
            - https://dblp.org/rdf/schema#creatorName

            - Available classes:
            - https://dblp.org/rdf/schema#Creator
            - https://dblp.org/rdf/schema#Person
            - https://dblp.org/rdf/schema#Publication
            - https://dblp.org/rdf/schema#Inproceedings
            - https://dblp.org/rdf/schema#Article
            - https://dblp.org/rdf/schema#Book
            - https://dblp.org/rdf/schema#AuthorSignature

            If you cannot generate a SPARQL query based on the provided examples, explain the reason to the user.

            Examples:
            {example}

            Question: {question}
            Author ID: {author_dblp_uri}
            Note: Do not include any explanations or apologies in your responses.
            Output only the Sparql query.
            Sparql:
        """
    else:
        for item in sim_questions[:examples]:
            sim_question = item['similar_question']
            sparql = utils.post_process_query(item['similar_question_sparql'])
            entities = ', '.join(item['entities'])
            example += (f"Question: {sim_question}\n"
                        f"Author ID: {entities}\n"
                        f"Sparql: {sparql}\n")
        return f"""
              Task: Generate SPARQL queries to query the DBLP knowledge graph based on the provided schema definition.
              Instructions:
              If you cannot generate a SPARQL query based on the provided examples, explain the reason to the user.
              {example}
              Question: {question}
              Author ID: {author_dblp_uri}
              Note: Do not include any explanations or apologies in your responses.
              Output only the Sparql query.
              Sparql:
            """


def generate_sparql(question, examples, with_schema):
    question_id = question['id']
    actual_question = question['question']
    author_dblp_uri = question['author_dblp_uri']
    prompt = construct_prompt(actual_question, question['top_n_similar_questions'], examples, author_dblp_uri, with_schema)
    print("Generated Prompt:", prompt)
    sparql = run_llm(prompt)
    cleaned_sparql = utils.post_process_query(sparql)
    cleaned_sparql = cleaned_sparql.replace('```sparql', '').replace('```', '').strip()
    result = {
        "id": question_id,
        "question": actual_question,
        "sparql": cleaned_sparql,
        "author_dblp_uri": author_dblp_uri,
        "given_answer": question['answer']
    }
    print("Generated SPARQL:", result)
    return result


def run_llm(prompt):
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found in environment variables.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    json_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 200
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_filename(base_path, with_schema, shot, limit, file_type):
    schema_part = "with_schema" if with_schema else "without_schema"
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    return f"{base_path}/{file_type}_{schema_part}_{shot}_shot_{limit}_Questions_{timestamp}.json"


def query_generation(top_n_similar_questions_path, base_save_path, shot, limit, with_schema):
    test_questions_similar_questions = utils.read_questions(top_n_similar_questions_path)
    save_generated_sparql_to = generate_filename(base_save_path, with_schema, shot, limit, "sparql_query")
    results = []
    for question in tqdm(test_questions_similar_questions[:limit], desc="Generating SPARQL queries"):
        result = generate_sparql(question, shot, with_schema)
        results.append(result)
    with open(save_generated_sparql_to, 'w') as outfile:
        json.dump(results, outfile, indent=2)
    print(f'Generated queries saved to {save_generated_sparql_to} successfully!')
    return save_generated_sparql_to


def write_predicted_answer_to_file(answer_results, file_name):
    newarr = []
    for item in answer_results:
        newanswer = []
        if not item['sparql_answer'] or 'results' not in item['sparql_answer']:
            newarr.append({"id": item["id"], "sparql_answer": [], "given_answer": item["given_answer"]})
        else:
            answer = item["sparql_answer"]
            if 'bindings' in answer['results']:
                for ans in answer['results']['bindings']:
                    for k, v in ans.items():
                        newanswer.append(ans[k]["value"])
                newarr.append({"id": item["id"], "sparql_answer": newanswer, "given_answer": item["given_answer"]})
    with open(file_name, 'w') as f:
        json.dump(newarr, f, indent=2)
    print('Saved to file!')


def error_analysis(answer_results, error_file):
    with open(error_file, 'w') as ef:
        for item in answer_results:
            if not item['sparql_answer']:
                ef.write(json.dumps(item) + "\n")
    print(f'Errors logged to {error_file}.')

def answer_extraction(query, max_retries=2):
    post_processed_query = utils.post_process_query(query)
    attempt = 0
    while attempt < max_retries:
        try:
            sparql = SPARQLWrapper("https://dblp-april24.skynet.coypu.org/sparql")
            # sparql = SPARQLWrapper("https://sparql.dblp.org/sparql")
            sparql.setQuery(post_processed_query)
            sparql.setReturnFormat(JSON)
            answer = sparql.query().convert()
            return answer
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {str(e)}")
            attempt += 1
    return None


def answer_generation(sparql_file_path, base_save_path, error_file_path):
    save_predicted_answer_to = sparql_file_path.replace("sparql_query", "answer")
    count = 0
    fin_queries = utils.read_questions(sparql_file_path)
    answer_results = []
    for query in tqdm(fin_queries, desc="Generating answers"):
        sparql_q = query['sparql']
        result = answer_extraction(sparql_q)
        if not result:
            count += 1
            print(f"No result for query id {query['id']}, attempt {count}")
        answer_results.append({
            'id': query['id'],
            'sparql_answer': result,
            'given_answer': query['given_answer'],
            'author_dblp_uri': query['author_dblp_uri']
        })
    write_predicted_answer_to_file(answer_results, save_predicted_answer_to)
    error_analysis(answer_results, error_file_path)
    print(f'Predicted answers saved to {save_predicted_answer_to} successfully!')


if __name__ == '__main__':
    top_n_similar_questions = "src/features/noise_reduction/generate_spaql/datasets/dev_questions_top_5_similar_questions.json"
    base_save_path = 'src/features/noise_reduction/generate_spaql/datasets/SPARQL'
    awnser_file_path = 'src/features/noise_reduction/generate_spaql/datasets/answers'
    error_file_path = 'src/features/noise_reduction/generate_spaql/datasets/failed_queries'
    shot = 3
    limit = 1000
    with_schema = False
    
    # sparql_query_path = query_generation(top_n_similar_questions, base_save_path, shot, limit, with_schema)
    sparql_query_path = "src/features/noise_reduction/generate_spaql/datasets/SPARQL/sparql_query_without_schema_3_shot_1000_Questions_20240712.json" 
    
    answer_generation(sparql_query_path, awnser_file_path, error_file_path)
    
    

