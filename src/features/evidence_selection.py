import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import BertTokenizer, BertModel, pipeline
import torch

# Initialize the Sentence-BERT model globally if not already done
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_triplet_extractor():
    return pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')


def extract_triple_from_question(question, triplet_extractor):
    """extract the triple from a question using llm

    Args:
        question (str): _description_

    Returns:
        set of triples: _description_
    """
    if (triplet_extractor == None):
        print("do not specify the triplet_extractor")
        exit(-1)
    extracted_texts = triplet_extractor.tokenizer.batch_decode([triplet_extractor(
        question, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
    L_extracted_triplets = []
    for extracted_text in extracted_texts:
        L_extracted_triplets += extract_triplets(extracted_text)
    return L_extracted_triplets


def extract_triplets(text):
    """
    Extract triplets from a text using predefined tags.

    This function processes a text input that contains triplets marked with specific tags
    ("<triplet>", "<subj>", "<obj>") and extracts these triplets into a structured format.

    Args:
        text (str): Input text containing marked triplets. Tags used:
                    - "<triplet>": Marks the beginning of a new triplet.
                    - "<subj>": Marks the subject of the triplet.
                    - "<obj>": Marks the object of the triplet.

    Returns:
        list of dict: A list of dictionaries, each representing a triplet extracted from the text.
                      Each dictionary has three keys:
                      - 'subject': The subject of the triplet.
                      - 'predicate': The relation or type of the triplet.
                      - 'object': The object of the triplet.
    """
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
    """select the sentence from quetion and sentences that

    Args:
        question (str): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    if (llm):
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
            q_embedding, sentences_embeddings, sentences, num_sentences=max(2, int(min(max_num, int(conserved_percentage*len(sentences)))/len(q_embeddings))))
    return evidence_sentences


def evidence_triple_selection(question, triples, conserved_percentage=0.1, max_num=50, triplet_extractor=None, llm=False):
    """select the triple match the question(directly compare question and triples)

    Args:
        question (str): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    if (llm):
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
    """selection of evidence sentences

    Args:
        target_embedding (): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    semantic_similarities = torch.tensor([model.similarity(
        triple_embedding, target_embedding) for triple_embedding in triples_embeddings])
    # if (threshold != None):
    #     semantic_similarities[semantic_similarities > threshold] = -1
    _, idx_list = torch.topk(semantic_similarities,
                             k=num_sentences, largest=True)

    return [triples[idx] for idx in idx_list if (semantic_similarities[idx] != -1)]


def evidence_sentence_selection_per_triple(triple, sentences, num_sentences=2):
    """selection of evidence sentences

    Args:
        triple (): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
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
    """Create embeddings from a sentence using Sentence-BERT model.

    Args:
        sentence (str): The input sentence from which to generate embeddings.

    Returns:
        torch.Tensor: A tensor containing the sentence embedding.
    """
    embedding = model.encode(sentence, convert_to_tensor=True)

    return embedding


def triple2text(triple):
    if (type(triple["object"]) == list):
        triple["object"] = '+'.join(triple["object"])
    results = triple["subject"] + " " + \
        triple["predicate"] + " " + triple["object"]
    return results


def create_embeddings_from_triple(triple):
    """Create embeddings from a textual triple using Sentence-BERT model.

    Args:
        triple (tuple): A tuple containing three elements (subject, predicate, object) which constitute the triple.

    Returns:
        torch.Tensor: The embedding representation of the concatenated triple.
    """
    # Convert the triple to a text string
    concatenated_triple = triple2text(triple)

    # Generate the embedding for the concatenated triple using Sentence-BERT
    # The model internally handles tokenization, pooling, and returns a sentence embedding
    embedding = model.encode(concatenated_triple, convert_to_tensor=True)
    return embedding


if __name__ == "__main__":
    with open("./data/processed/DBLP_first_10Q.json") as f:
        data = json.load(f)[0]
    ems = evidence_triple_selection(
        data['question'], data['all_tripples'], conserved_percentage=0.1)
    print(len(ems))
