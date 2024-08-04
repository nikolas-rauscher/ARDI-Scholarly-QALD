# coding: utf-8
import json
import numpy as np
from data.evidence_selection import evidence_sentence_selection, load_triplet_extractor
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, pipeline
import torch

def test_evidence_selection_wiki():
    with open("./data/external/preprocessed_full_dataset_wiki.json") as f:
        data = json.load(f)[6]
    
    extract_triplet = load_triplet_extractor()
    
    for wiki_text in data['wiki_data']:
        for key, item in wiki_text.items():
            selected_evidence = evidence_sentence_selection(
                data['question'], item.split('.'), triplet_extractor=extract_triplet, llm=True, conserved_percentage=0.1)
            assert selected_evidence is not None  # Basic check to ensure function returns a result
            # Additional assertions can be added here based on expected outcomes

if __name__ == "__main__":
    test_evidence_selection_wiki()
