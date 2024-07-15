from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss
import os

def create_faiss_index(example_questions_file, index_file, embeddings_file, max_examples=10000):
    if os.path.exists(embeddings_file):
        print("Loading saved embeddings...")
        embeddings = np.load(embeddings_file)
        example_questions = json.load(open(example_questions_file, 'r'))[:max_examples]
    else:
        print("Loading example questions...")
        # Load example questions
        example_questions = json.load(open(example_questions_file, 'r'))[:max_examples]

        print("Loading the sentence transformer model...")
        # Load the sentence transformer model
        model = SentenceTransformer('all-mpnet-base-v2')

        print(f"Creating embeddings for the first {max_examples} questions...")
        # Encode both the main and alternative questions for the example questions
        main_embeddings = model.encode([q['question'] for q in example_questions])
        
        # Handling case where 'paraphrased_question' might be empty
        alt_embeddings = model.encode([q['paraphrased_question'] if q['paraphrased_question'] else q['question'] for q in example_questions])

        # Combine main and alternative embeddings
        embeddings = np.vstack((main_embeddings, alt_embeddings))

        print("Saving embeddings to a file...")
        np.save(embeddings_file, embeddings)

    print("Creating and populating the FAISS index...")
    # Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    print("Saving the index to a file...")
    # Save the index
    faiss.write_index(index, index_file)

    print(f"Number of vectors in the index: {index.ntotal}")

    return example_questions

def find_similar_questions(input_questions_file, example_questions, index_file, output_file, n=5):
    print("Loading input questions and FAISS index...")
    # Load input questions and FAISS index
    input_questions = json.load(open(input_questions_file, 'r'))#[:2]  # Only the first 2 questions
    index = faiss.read_index(index_file)

    print(f"Number of vectors in the index: {index.ntotal}")

    print("Loading the sentence transformer model...")
    # Load the sentence transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Creating embeddings for input questions...")
    # Encode the input questions
    input_question_embeddings = model.encode([q['question'] for q in input_questions])

    print("Finding the most similar questions...")
    # Find the most similar questions for each input question
    output_data = []
    for i, question_embedding in enumerate(input_question_embeddings):
        print(f"Processing question {i + 1}/{len(input_question_embeddings)}...")
        # Search for similar questions in the FAISS index
        D, I = index.search(question_embedding.reshape(1, -1).astype('float32'), 2 * n)
        
        # Select the top n unique questions
        selected_indices = []
        seen_questions = set()
        for idx in I[0]:
            question_idx = idx % len(example_questions)
            question_id = example_questions[question_idx]['id']
            if question_id not in seen_questions and len(selected_indices) < n:
                selected_indices.append(idx)
                seen_questions.add(question_id)

        # Get the corresponding questions and SPARQL queries from the example questions
        similar_questions = []
        for idx in selected_indices:
            question_idx = idx % len(example_questions)
            similar_q = example_questions[question_idx]
            similar_question_data = {
                "id": similar_q['id'],
                "similar_question": similar_q['question'],
                "alternative_question": similar_q['paraphrased_question'] if similar_q['paraphrased_question'] else similar_q['question'],
                "similar_question_sparql": similar_q['query'],
                "question_type": similar_q['query_type'],
                "matched_question": "main" if idx < len(example_questions) else "alternative",
                "entities": similar_q.get('entities', [])
            }
            similar_questions.append(similar_question_data)

        # Create the output structure for this question
        question_data = {
            "id": input_questions[i]['id'],
            "question": input_questions[i]['question'],
            "author_dblp_uri": input_questions[i].get('author_dblp_uri', ''),
            "answer": input_questions[i].get('answer', ''),
            "top_n_similar_questions": similar_questions
        }
        output_data.append(question_data)

    print("Saving the output data to a JSON file...")
    # Save the output data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=3)

    print("Done!")

# Usage:
print("Creating the FAISS index...")
example_questions = create_faiss_index(
    example_questions_file="input_questions_v2.json",
    index_file="embeddings/example_questions_index_all-mpnet-base-v2.faiss",
    embeddings_file="embeddings/example_questions_embeddings_all-mpnet-base-v2.npy",
    max_examples=10000
)

print("Finding the top n similar questions...")
find_similar_questions(
    input_questions_file="train_dataset.json",
    example_questions=example_questions,
    index_file="embeddings/example_questions_index_all-mpnet-base-v2.faiss",
    output_file="dev_questions_top_5_similar_questions_v2.json",
    n=5
)

print("Process completed.")