import os
from tqdm import tqdm
import configparser
import json
from models.verbalizer.generatePrompt import verbalise_triples
from features.evidence_selection import evidence_triple_selection, triple2text, evidence_sentence_selection
import sys
import torch

# Set device (CPU or GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Define the root path of the project
ROOT_PATH = "/content/drive/MyDrive/repo/ARDI-Scholarly-QALD"

# Add the root path to sys.path
sys.path.append(ROOT_PATH)

# Set the config file path
CONFIG_PATH = os.path.join(ROOT_PATH, 'config.ini')

# Print the root path for verification
print(ROOT_PATH)

# Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# Ensure the model is loaded to the correct device
model = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

def prepare_data_only_ve(examples, prompt_template, output_file):
    """
    Prepare the data by generating contexts for each example with all 3 resources

    Args:
        examples (list): List of examples.
        prompt_template (str): Prompt template.
        output_file (str): Output file path.

    Returns:
        None
    """
    prepared_data = []

    for example in tqdm(examples, desc="Preparing Data"):
        # Number of triples
        triples_number = len(example['all_triples'])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples']
        )

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        wiki_context = ""
        for wiki_text in example['wiki_data']:
            if len(wiki_text) > 0:
                sentences = str(wiki_text).split('.')
                wiki_evidence = evidence_sentence_selection(
                    example['question'], sentences, conserved_percentage=0.2, max_num=40
                )
                wiki_context += '. '.join(wiki_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": context_evidence_verbalizer + wiki_context
        }

        if "answer" in example:
            prepared_example["answer"] = example["answer"]

        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)

def prepare_data_4settings(examples, prompt_template, output_file, wiki=True):
    """
    Prepare the data by generating contexts for each example with dnlp and openalex

    Args:
        examples (list): List of examples.
        prompt_template (str): Prompt template.
        output_file (str): Output file path.

    Returns:
        None
    """
    prepared_data = []
    for example in tqdm(examples, desc="Preparing Data"):
        triples_number = len(example['all_triples'])

        # wiki
        wiki_context = ""
        wiki_context_plain = ""
        if wiki:
            for wiki_text in example['wiki_data']:
                wiki_context_plain += str(wiki_text)
                if len(wiki_text) > 0:
                    sentences = str(wiki_text).split('.')
                    wiki_evidence = evidence_sentence_selection(
                        example['question'], sentences, conserved_percentage=0.2, max_num=40
                    )
                    wiki_context += '. '.join(wiki_evidence)

        # Plain Triples
        context_plain = '. '.join([triple2text(triple) for triple in example['all_triples']])

        # Evidence Matching
        triples_evidence = evidence_triple_selection(
            example['question'], example['all_triples'][0])
        context_evidence = '. '.join([triple2text(triple) for triple in triples_evidence])

        # Verbalizer
        context_verbalizer = verbalise_triples(example['all_triples'])

        # Verbalizer + Evidence Matching
        context_evidence_verbalizer = verbalise_triples(triples_evidence)

        prepared_example = {
            "id": example["id"],
            "question": example["question"],
            "triples_number": triples_number,
            "contexts": {
                "all_triples": example['all_triples'],
                "plain": context_plain + wiki_context_plain,
                "verbalizer_on_all_triples": context_verbalizer + wiki_context_plain,
                "evidence_matching": context_evidence + wiki_context,
                "verbalizer_plus_evidence_matching": context_evidence_verbalizer + wiki_context
            }
        }
        
        if "answer" in example:
            prepared_example["answer"] = example["answer"]
        prepared_data.append(prepared_example)

    with open(output_file, 'w') as file:
        json.dump(prepared_data, file, indent=4, ensure_ascii=False)

def process_file(input_file_path, prompt_template_path, output_file_path):
    """
    Process the input file and generate prepared data.

    Args:
        input_file_path (str): Input file path.
        prompt_template_path (str): Prompt template file path.
        output_file_path (str): Output file path.

    Returns:
        bool: True if the process is successful, False otherwise.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        return False
    if not os.path.exists(prompt_template_path):
        print(f"Error: Prompt template file '{prompt_template_path}' not found.")
        return False

    with open(input_file_path, 'r') as file:
        examples = json.load(file)
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    prepare_data_only_ve(examples, prompt_template, output_file_path)
    return True

if __name__ == '__main__':
    with open(config['FilePaths']['prepare_prompt_context_input'], 'r') as file:
        examples = json.load(file)
    with open(config['FilePaths']['prompt_template'], 'r') as file:
        prompt_template = file.read()

    output_file = config['FilePaths']['prepared_data_file']
    prepare_data_only_ve(examples, prompt_template, output_file)

# Additional functions and logic

def load_triplet_extractor():
    return pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

def extract_triple_from_question(question, triplet_extractor):
    if triplet_extractor is None:
        print("do not specify the triplet_extractor")
        exit(-1)
    extracted_texts = triplet_extractor.tokenizer.batch_decode([triplet_extractor(
        question, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
    L_extracted_triplets = []
    for extracted_text in extracted_texts:
        L_extracted_triplets += extract_triplets(extracted_text)
    return L_extracted_triplets

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append(
            {'subject': subject.strip(), 'predicate': relation.strip(), 'object': object_.strip()})
    return triplets

def evidence_sentence_selection(question, sentences, conserved_percentage=0.1, max_num=50, triplet_extractor=None, llm=False):
    if llm:
        q_embeddings = [create_embeddings_from_sentence(
            sentence) for sentence in extract_triple_from_question(question, triplet_extractor)]
    else:
        q_embeddings = [create_embeddings_from_sentence(
            question)]
    evidence_sentences = []
    sentences_embeddings = [create_embeddings_from_sentence(
        sentence) for sentence in sentences]
    for q_embedding in q_embeddings:
        evidence_sentences += evidence_selection_per_embedding(
            q_embedding, sentences_embeddings, sentences, num_sentences=max(2, int(min(max_num, int(conserved_percentage * len(sentences))) / len(q_embeddings))))
    return evidence_sentences

def evidence_triple_selection(question, triples, conserved_percentage=0.1, max_num=50, triplet_extractor=None, llm=False):
    if llm:
        q_embeddings = [create_embeddings_from_triple(
            triple) for triple in extract_triple_from_question(question, triplet_extractor)]
    else:
        q_embeddings = [create_embeddings_from_sentence(question)]
    evidence_triples = []
    triples_embeddings = [create_embeddings_from_triple(
        triple) for triple in triples]
    for q_embedding in q_embeddings:
        evidence_triples += evidence_selection_per_embedding(
            q_embedding, triples_embeddings, triples, num_sentences=min(max_num, int(conserved_percentage * len(triples))))
    return evidence_triples

def evidence_selection_per_embedding(target_embedding, triples_embeddings, triples, num_sentences=2):
    semantic_similarities = torch.tensor([model.similarity(
        triple_embedding, target_embedding) for triple_embedding in triples_embeddings]).to(DEVICE)  # Ensure similarities are on the same device
    _, idx_list = torch.topk(semantic_similarities,
                             k=num_sentences, largest=True)
    return [triples[idx] for idx in idx_list if semantic_similarities[idx] != -1]

def evidence_sentence_selection_per_triple(triple, sentences, num_sentences=2):
    triple_embedding = create_embeddings_from_triple(triple)
    sentence_embeddings = torch.tensor(
        [create_embeddings_from_sentence(sentence) for sentence in sentences]).to(DEVICE)
    semantic_similarities = torch.tensor([model.similarity(
        triple_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]).to(DEVICE)
    _, idx_list = torch.topk(semantic_similarities,
                             k=num_sentences, largest=True)
    return [sentences[idx] for idx in idx_list]

def create_embeddings_from_sentence(sentence):
    embedding = model.encode(sentence, convert_to_tensor=True).to(DEVICE)
    return embedding

def triple2text(triple):
    if isinstance(triple["object"], list):
        triple["object"] = '+'.join(triple["object"])
    return f'{triple["subject"]} {triple["predicate"]} {triple["object"]}'

def create_embeddings_from_triple(triple):
    concatenated_triple = triple2text(triple)
    embedding = model.encode(concatenated_triple, convert_to_tensor=True).to(DEVICE)
    return embedding

if __name__ == "__main__":
    with open("./data/processed/DBLP_first_10Q.json") as f:
        data = json.load(f)[0]
    ems = evidence_triple_selection(
        data['question'], data['all_triples'], conserved_percentage=0.1)
    print(len(ems))