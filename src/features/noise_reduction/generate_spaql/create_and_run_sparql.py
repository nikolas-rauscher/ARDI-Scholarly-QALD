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
    """
    Constructs a prompt for generating SPARQL queries based on the provided question, similar questions, examples, and author DBLP URI.

    Args:
        question (str): The question to generate a SPARQL query for.
        sim_questions (list): A list of similar questions.
        examples (int): The number of examples to include in the prompt.
        author_dblp_uri (str): The DBLP URI of the author.
        with_schema (bool, optional): Whether to include the schema definition in the prompt. Defaults to False.

    Returns:
        str: The constructed prompt.
    """
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


def generate_sparql(question, examples, with_schema, model_name):
    """
    Generates a SPARQL query based on the provided question, examples, and model name.

    Args:
        question (dict): The question to generate a SPARQL query for.
        examples (int): The number of examples to include in the prompt.
        with_schema (bool): Whether to include the schema definition in the prompt.
        model_name (str): The name of the model to use for generating the SPARQL query.

    Returns:
        dict: A dictionary containing the question ID, question, generated SPARQL query, author DBLP URI, and given answer.
    """
    question_id = question['id']
    actual_question = question['question']
    author_dblp_uri = question['author_dblp_uri']
    prompt = construct_prompt(actual_question, question['top_n_similar_questions'], examples, author_dblp_uri, with_schema)
    print("Generated Prompt:", prompt)
    sparql = run_llm(prompt, model_name)
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


def run_llm(prompt, model_name):
    """
    Runs a Large Language Model (LLM) to generate a SPARQL query based on the provided prompt and model name.

    Args:
        prompt (str): The prompt to generate a SPARQL query for.
        model_name (str): The name of the model to use for generating the SPARQL query.

    Returns:
        str: The generated SPARQL query.
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found in environment variables.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    json_data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 200
    }
    
    max_retries = 5
    backoff_factor = 0.3
    for attempt in range(max_retries):
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data, timeout=(10, 60))
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                raise


def generate_filename(base_path, with_schema, shot, limit, model_name, file_type):
    """
    Generates a filename based on the provided parameters.

    Args:
        base_path (str): The base path for the file.
        with_schema (bool): Whether the schema definition is included.
        shot (int): The number of shots (examples) used.
        limit (int): The limit of questions.
        model_name (str): The name of the model used.
        file_type (str): The type of file (e.g., "sparql_query" or "answer").

    Returns:
        str: The generated filename.
    """
    schema_part = "with_schema" if with_schema else "without_schema"
    # timestamp = datetime.datetime.now().strftime("%Y%m%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_path}/{file_type}_{schema_part}_{shot}_shot_{limit}_Questions_{model_name}_{timestamp}.json"


def query_generation(top_n_similar_questions_path, base_save_path, shot, limit, with_schema, model_name):
    """
    Generates SPARQL queries for a list of questions and saves them to a file.

    Args:
        top_n_similar_questions_path (str): The path to the file containing the top N similar questions.
        base_save_path (str): The base path to save the generated SPARQL queries.
        shot (int): The number of shots (examples) used.
        limit (int): The limit of questions.
        with_schema (bool): Whether the schema definition is included.
        model_name (str): The name of the model used.

    Returns:
        str: The path to the file where the generated SPARQL queries are saved.
    """
    test_questions_similar_questions = utils.read_questions(top_n_similar_questions_path)
    save_generated_sparql_to = generate_filename(base_save_path, with_schema, shot, limit, model_name, "sparql_query")
    results = []
    for question in tqdm(test_questions_similar_questions[:limit], desc="Generating SPARQL queries"):
        result = generate_sparql(question, shot, with_schema, model_name)
        results.append(result)
    with open(save_generated_sparql_to, 'w') as outfile:
        json.dump(results, outfile, indent=2)
    print(f'Generated queries saved to {save_generated_sparql_to} successfully!')
    return save_generated_sparql_to


def write_predicted_answer_to_file(answer_results, file_name):
    """
    Writes the predicted answers to a file.

    Args:
        answer_results (list): A list of dictionaries containing the question ID, SPARQL answer, and given answer.
        file_name (str): The name of the file to write the answers to.
    """
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
    """
    Analyzes errors in the answer results and logs them to a file.

    Args:
        answer_results (list): A list of dictionaries containing the question ID, SPARQL answer, and given answer.
        error_file (str): The name of the file to log errors to.
    """
    with open(error_file, 'w') as ef:
        for item in answer_results:
            if not item['sparql_answer']:
                ef.write(json.dumps(item) + "\n")
    print(f'Errors logged to {error_file}.')

def answer_extraction(query, max_retries=2):
    """
    Extracts answers from the DBLP knowledge graph based on the provided SPARQL query.

    Args:
        query (str): The SPARQL query to execute.
        max_retries (int, optional): The maximum number of retries in case of failure. Defaults to 2.

    Returns:
        dict or None: The extracted answers or None if failed.
    """
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
    """
    Generates answers for the SPARQL queries and saves them to a file.

    Args:
        sparql_file_path (str): The path to the file containing the SPARQL queries.
        base_save_path (str): The base path to save the generated answers.
        error_file_path (str): The path to the file to log errors.
    """
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
    error_file_path = 'src/features/noise_reduction/generate_spaql/datasets/failed_queries/failed_queries.json'
    shot = 5
    limit = 10000
    with_schema = True
    model_name = "gpt-3.5-turbo"
    
    sparql_query_path = query_generation(top_n_similar_questions, base_save_path, shot, limit, with_schema, model_name)
    # sparql_query_path = "src/features/noise_reduction/generate_spaql/datasets/SPARQL/sparql_query_without_schema_3_shot_1000_Questions_20240712.json" 
    
    answer_generation(sparql_query_path, awnser_file_path, error_file_path)