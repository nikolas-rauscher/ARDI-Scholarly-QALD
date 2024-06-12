from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
extract_triple_prompt = ""


def extract_triple_from_question(question):
    """extract the triple from a question using llm

    Args:
        question (str): _description_

    Returns:
        set of triples: _description_
    """
    triples = set()
    return triples


def evidence_sentence_selection(question, sentences, num_sentences=2):
    """select the sentence from quetion and sentences that

    Args:
        question (str): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    triples = extract_triple_from_question(question)
    sentences = []
    for triple in triples:
        sentences += evidence_sentence_selection_per_triple(
            triple, sentences, num_sentences=num_sentences)
    return sentences


def evidence_triple_selection(question, triples, num_triples=2, llm=False):
    """select the triple match the question(directly compare question and triples)

    Args:
        question (str): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    if (llm):
        q_embeddings = [create_embeddings_from_triple(
            triple) for triple in extract_triple_from_question(question)]
    else:
        q_embeddings = [create_embeddings_from_sentence(question)]
    evidence_triples = []
    triples_embeddings = [create_embeddings_from_triple(
        triple) for triple in triples]
    print("finished creating embeddings")
    for q_embedding in q_embeddings:
        evidence_triples += evidence_triple_selection_per_embedding(
            q_embedding, triples_embeddings, triples, num_sentences=num_triples)
    return evidence_triples


def evidence_triple_selection_per_embedding(target_embedding, triples_embeddings, triples, num_sentences=2):
    """selection of evidence sentences

    Args:
        target_embedding (): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    semantic_similarities = torch.tensor([torch.dist(
        triple_embedding, target_embedding, p=2).item() for triple_embedding in triples_embeddings])
    _, idx_list = torch.topk(-semantic_similarities,
                             k=num_sentences, largest=True)
    return [triples[idx] for idx in idx_list]


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
    semantic_similarities = torch.tensor([torch.dist(
        triple_embedding, sentence_embedding, p=2) for sentence_embedding in sentence_embeddings])
    _, idx_list = torch.topk(-semantic_similarities,
                             k=num_sentences, largest=True)
    return sentences[idx_list]


def create_embeddings_from_sentence(s):
    """create embeddings from sentece
    Args:
        s (str): _description_

    Returns:
        embedding: _description_
    """
    global tokenizer
    global model
    inputs = tokenizer(s, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']

    with torch.no_grad():  # No need to compute gradients for embedding extraction
        output = model(**inputs)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = output.last_hidden_state

    return last_hidden_state[:, 0, :].squeeze()


def create_embeddings_from_triple(triple):
    """create embeddings from a textual triple

    Args:
        triple (tuple): A tuple containing three elements (subject, predicate, object) which constitute the triple.

    Returns:
        embedding: The embedding representation of the concatenated triple.
    """
    global tokenizer
    global model
    # Concatenate the elements of the triple with appropriate separators
    concatenated_triple = f"{triple["subject"]} [SEP] {triple["predicate"]} [SEP] {triple["object"]}"

    # Tokenize the concatenated triple
    inputs = tokenizer(concatenated_triple, return_tensors='pt',
                       padding=True, truncation=True)
    input_ids = inputs['input_ids']

    with torch.no_grad():  # No need to compute gradients for embedding extraction
        output = model(**inputs)
        # (batch_size, sequence_length, hidden_size)
        last_hidden_state = output.last_hidden_state

    # Extract the embeddings corresponding to the '[CLS]' token or another meaningful aggregation
    return last_hidden_state[:, 0, :].squeeze()
