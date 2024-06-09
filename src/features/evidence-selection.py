from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
extract_triple_prompt=""

def extract_triple_from_question(question):
    """extract the triple from a question using llm

    Args:
        question (str): _description_

    Returns:
        set of triples: _description_
    """
    triples = set()
    return triples


def evidence_sentence_selection(triple, sentences, num_sentences=2):
    """selection of evidence sentences

    Args:
        triple (): _description_
        sentences (str[]): _description_
        num_sentences (int, optional): _description_. Defaults to 2.
    """
    triple_embedding = create_embeddings_from_triple(triple)
    sentence_embeddings = torch.tensor(
        [create_embeddings_from_triple(sentence) for sentence in sentences])
    semantic_similarities = torch.tensor([torch.dist(
        triple_embedding, sentence_embedding, p=2) for sentence_embedding in sentence_embeddings])
    _, idx_list = torch.topk(-semantic_similarities,
                             k=num_sentences, largest=True)
    return sentences[idx_list]


def create_embeddings_from_triple(triple):
    """create an embedding from a triple

    Args:
        triple (str): _description_

    Returns:
        embedding: _description_
    """
    global tokenizer
    global model


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
