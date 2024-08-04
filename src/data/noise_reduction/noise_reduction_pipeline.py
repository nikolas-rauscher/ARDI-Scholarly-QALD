import configparser
import logging
import os
from dotenv import load_dotenv
from data.noise_reduction.utils import download_dataset
from data.noise_reduction.crate_top_n import create_faiss_index, find_similar_questions
from data.noise_reduction.simple_noise_reduction import clean_and_save_dataset
from data.noise_reduction.create_and_run_sparql import query_generation, answer_generation

# Konfiguration einlesen
config = configparser.ConfigParser()
config.read("config.ini")

# .env Datei laden
load_dotenv()

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_api_key():
    """
    Checks if the OPENAI_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is set, False otherwise.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY is missing in .env file.")
        return False
    return True

def main():
    logging.info("Starting the data processing workflow...")

    # Lade die notwendigen Pfade aus der Konfiguration
    data_folder = config['FilePaths']['noise_reduction_data_folder']
    raw_train_dataset = config['FilePaths']['raw_questions_path']
    dataset_path = f"{data_folder}/2023_dataset.json"

    # Antwort- und Fehlerdateipfade relativ zu dataset_path setzen
    answer_file_path = os.path.join(os.path.dirname(dataset_path), "answers")
    error_file_path = os.path.join(os.path.dirname(dataset_path), "failed_queries", "failed_queries.json")

    # Ordner erstellen, falls sie nicht existieren
    embeddings_dir = f"{data_folder}/embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(answer_file_path, exist_ok=True)
    os.makedirs(os.path.dirname(error_file_path), exist_ok=True)  # Verzeichnis für Fehlerdateien erstellen

    # Parameter aus dem Abschnitt noise_reduction_parameters laden
    shot = int(config['noise_reduction_parameters']['shot'])
    limit = int(config['noise_reduction_parameters']['limit'])
    with_schema = config.getboolean('noise_reduction_parameters', 'with_schema')
    model_name = config['noise_reduction_parameters']['model_name']

    # Herunterladen des Datensatzes
    download_dataset(dataset_path)

    # Rauschreduzierung durchführen
    logging.info("Reducing noise in the dataset...")
    clean_dataset_path = f"{data_folder}/clean_dataset.json"
    clean_and_save_dataset(raw_train_dataset, clean_dataset_path)

    # FAISS-Index erstellen
    logging.info("Creating the FAISS index...")
    index_file = f"{embeddings_dir}/example_question_index.faiss"
    embeddings_file = f"{embeddings_dir}/example_questions_embeddings.npy"
    example_questions = create_faiss_index(
        example_questions_file=dataset_path,
        index_file=index_file,
        embeddings_file=embeddings_file,
        max_examples=10
    )

    # Ähnliche Fragen finden
    logging.info("Finding the top n similar questions...")
    similar_questions_file = f"{embeddings_dir}/dev_questions_top_5_similar_questions.json"
    find_similar_questions(
        input_questions_file=clean_dataset_path,  # Verwende den Pfad der gereinigten Daten
        example_questions=example_questions,
        index_file=index_file,
        output_file=similar_questions_file,
        n=5
    )

    # Überprüfe, ob der API-Schlüssel vorhanden ist
    if check_api_key():
        # Verzeichnis für SPARQL-Abfragen erstellen
        base_save_path = f"{data_folder}/SPARQL"
        os.makedirs(base_save_path, exist_ok=True)

        logging.info("Generating SPARQL queries and extracting answers...")
        sparql_query_path = query_generation(similar_questions_file, base_save_path, shot, limit, with_schema, model_name)
        answer_generation(sparql_query_path, answer_file_path, error_file_path)
    else:
        logging.warning("Skipping SPARQL query generation and answer extraction due to missing API key.")

    logging.info("Process completed. Output stored in specified files.")

if __name__ == "__main__":
    main()