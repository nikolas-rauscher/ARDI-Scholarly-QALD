import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import BertTokenizer, BertModel, pipeline
import torch

# Initialize the Sentence-BERT model globally if not already done
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_triplet_extractor():
    """Loads a triplet extraction model pipeline.

    Returns:
        A pipeline for extracting triples from text using a specified model and tokenizer.
    """
    return pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')


def extract_triple_from_question(question, triplet_extractor):
    """Extracts triples from a question using a large language model (LLM).

    Args:
        question: The input question from which to extract triples.
        triplet_extractor: A pipeline for extracting triples.

    Returns:
        A list of triples extracted from the question, where each triple is represented as a dictionary.
    """
    if triplet_extractor is None:
        print("Triplet extractor is not specified.")
        exit(-1)

    extracted_texts = triplet_extractor.tokenizer.batch_decode([triplet_extractor(
        question, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
    L_extracted_triplets = []
    for extracted_text in extracted_texts:
        L_extracted_triplets += extract_triplets(extracted_text)
    return L_extracted_triplets


def extract_triplets(text):
    """Extracts triplets from a text using predefined tags.

    Args:
        text: Input text containing marked triplets. Tags used:
                    - "<triplet>": Marks the beginning of a new triplet.
                    - "<subj>": Marks the subject of the triplet.
                    - "<obj>": Marks the object of the triplet.

    Returns:
        A list of dictionaries, each representing a triplet extracted from the text.
                      Each dictionary has three keys:
                      - 'subject': The subject of the triplet.
                      - 'predicate': The relation or type of the triplet.
                      - 'object': The object of the triplet.
    """
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(
                ), 'predicate': relation.strip(), 'object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(
                ), 'predicate': relation.strip(), 'object': object_.strip()})
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
        triplets.append({'subject': subject.strip(
        ), 'predicate': relation.strip(), 'object': object_.strip()})
    return triplets


def evidence_sentence_selection(question, sentences, conserved_percentage=0.1, max_num=50, triplet_extractor=None, llm=False):
    """Selects relevant sentences based on the input question and a list of sentences.

    Args:
        question: The question used to select relevant sentences.
        sentences: A list of sentences from which to select relevant ones.
        conserved_percentage: The percentage of sentences to conserve. Defaults to 0.1.
        max_num: The maximum number of sentences to select. Defaults to 50.
        triplet_extractor: The triplet extraction model. Defaults to None.
        llm: Whether to use LLM-based triplet extraction. Defaults to False.

    Returns:
        A list of selected sentences that are relevant to the question.
    """
    if llm:
        q_embeddings = [create_embeddings_from_sentence(
            sentence) for sentence in extract_triple_from_question(question, triplet_extractor)]
    else:
        q_embeddings = [create_embeddings_from_sentence(question)]

    evidence_sentences = []
    sentences_embeddings = [create_embeddings_from_sentence(
        sentence) for sentence in sentences]
    for q_embedding in q_embeddings:
        evidence_sentences += evidence_selection_per_embedding(
            q_embedding, sentences_embeddings, sentences, num_sentences=max(2, int(min(max_num, int(conserved_percentage*len(sentences)))/len(q_embeddings))))
    return evidence_sentences


def evidence_triple_selection(question, triples, conserved_percentage=0.1, max_num=50, triplet_extractor=None, llm=False):
    """Selects relevant triples based on the input question and a list of triples.

    Args:
        question: The question used to select relevant triples.
        triples: A list of triples from which to select relevant ones.
        conserved_percentage: The percentage of triples to conserve. Defaults to 0.1.
        max_num: The maximum number of triples to select. Defaults to 50.
        triplet_extractor: The triplet extraction model. Defaults to None.
        llm: Whether to use LLM-based triplet extraction. Defaults to False.

    Returns:
        A list of selected triples that are relevant to the question.
    """
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
            q_embedding, triples_embeddings, triples, num_sentences=min(max_num, int(conserved_percentage*len(triples))))
    return evidence_triples


def evidence_selection_per_embedding(target_embedding, triples_embeddings, triples, num_sentences=2):
    """Selects the most semantically similar triples based on a target embedding.

    Args:
        target_embedding: The embedding of the target (question or triple).
        triples_embeddings: A list of embeddings for the triples.
        triples: A list of triples corresponding to the embeddings.
        num_sentences: The number of triples to select. Defaults to 2.

    Returns:
        A list of selected triples that are most semantically similar to the target.
    """
    semantic_similarities = torch.tensor([model.similarity(
        triple_embedding, target_embedding) for triple_embedding in triples_embeddings])
    _, idx_list = torch.topk(semantic_similarities,
                             k=num_sentences, largest=True)
    return [triples[idx] for idx in idx_list if semantic_similarities[idx] != -1]


def evidence_sentence_selection_per_triple(triple, sentences, num_sentences=2):
    """Selects the most semantically similar sentences based on a triple.

    Args:
        triple: The input triple used to select relevant sentences.
        sentences: A list of sentences from which to select relevant ones.
        num_sentences: The number of sentences to select. Defaults to 2.

    Returns:
        A list of selected sentences that are most semantically similar to the triple.
    """
    triple_embedding = create_embeddings_from_triple(triple)
    sentence_embeddings = torch.tensor(
        [create_embeddings_from_sentence(sentence) for sentence in sentences])
    semantic_similarities = torch.tensor([model.similarity(
        triple_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings])
    _, idx_list = torch.topk(semantic_similarities,
                             k=num_sentences, largest=True)
    return sentences[idx_list]

def create_embeddings_from_sentence(sentence):
    """Create embeddings from a sentence using the Sentence-BERT model.

    Args:
        sentence: The input sentence from which to generate embeddings.

    Returns:
        A tensor containing the sentence embedding.
    """
    embedding = model.encode(sentence, convert_to_tensor=True)
    return embedding


def triple2text(triple):
    """Converts a triple dictionary into a text string.

    Args:
        triple: A dictionary containing 'subject', 'predicate', and 'object' of the triple.

    Returns:
        A string representation of the triple.
    """
    if isinstance(triple["object"], list):
        triple["object"] = '+'.join(triple["object"])
    return f"{triple['subject']} {triple['predicate']} {triple['object']}"


def create_embeddings_from_triple(triple):
    """Create embeddings from a textual triple using the Sentence-BERT model.

    Args:
        triple: A dictionary containing 'subject', 'predicate', and 'object' of the triple.

    Returns:
        The embedding representation of the concatenated triple.
    """
    concatenated_triple = triple2text(triple)
    embedding = model.encode(concatenated_triple, convert_to_tensor=True)
    return embedding

if __name__ == "__main__":
    with open("./data/processed/prepared_data_10q.json") as f:
        data = json.load(f)[6]
    extract_triplet = load_triplet_extractor()
    ems = evidence_triple_selection(
        data['question'], data['contexts']['all_triples'], triplet_extractor=extract_triplet, llm=False, conserved_percentage=0.1)
    print(data['triples_number'], len(ems))
