import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load pre-trained BERT model for semantic similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

def calculate_similarity(answer1, answer2):
    try:
        if answer1.strip() and answer2.strip():
            # Encode the answers
            embeddings1 = model.encode(answer1, convert_to_tensor=True)
            embeddings2 = model.encode(answer2, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])  
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculating similarity between '{answer1}' and '{answer2}': {e}")
        return 0.0

def compare_answers(data):
    results = []
    for item in data:
        sparql_answers = item['sparql_answer']
        given_answer = item['given_answer']
        
        if sparql_answers and given_answer.strip():
            similarities = [calculate_similarity(sparql_answer, given_answer) for sparql_answer in sparql_answers]
            max_similarity = max(similarities) if similarities else 0.0
        else:
            max_similarity = 0.0

        results.append({
            "id": item["id"],
            "sparql_answers": sparql_answers,
            "given_answer": given_answer,
            "similarity": max_similarity
        })

        print(f"Processed item {item['id']}: max_similarity={max_similarity}")

    return results

def summary_statistics(results):
    similarities = [item["similarity"] for item in results if item["similarity"] > 0]
    num_given_answers = len([item for item in results if item["given_answer"].strip()])
    num_sparql_answers = sum(len(item["sparql_answers"]) for item in results)
    num_sparql_answers_non_zero = sum(len([answer for answer in item["sparql_answers"] if answer != "0"]) for item in results)

    summary = {
        "total_questions": len(results),
        "num_given_answers": num_given_answers,
        "num_sparql_answers": num_sparql_answers,
        "num_sparql_answers_non_zero": num_sparql_answers_non_zero,
        "average_similarity": np.mean(similarities) if similarities else 0,
        "max_similarity": np.max(similarities) if similarities else 0,
        "min_similarity": np.min(similarities) if similarities else 0
    }

    return summary

def save_results(results, summary, output_file):
    with open(output_file, 'w') as file:
        json.dump({"results": results, "summary": summary}, file, indent=2)
    print(f'Results saved to {output_file} successfully!')

def process_directory(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.startswith("answer_"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, filename.replace(".json", "_ergebnisse.json"))
            
            data = load_json(input_file)
            results = compare_answers(data)
            summary = summary_statistics(results)
            save_results(results, summary, output_file)

            print(f"Processed file: {filename}")

if __name__ == "__main__":
    input_directory = "src/features/noise_reduction/generate_spaql/datasets/answers"  
    output_directory = "src/features/noise_reduction/generate_spaql/datasets/eval_results"  

    process_directory(input_directory, output_directory)