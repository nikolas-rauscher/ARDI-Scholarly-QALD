from datasets import load_dataset
import json

def download_dataset(output_file):
    """
    Downloads a dataset and saves it to a specified output file.
    
    Parameters:
        output_file (str): The path to the output file where the dataset will be saved.
    Returns:
        None
    """
    ds = load_dataset("awalesushil/DBLP-QuAD")
    new_data = []

    for item in ds['train']:
        new_data.append({
            "id": item["id"],
            "question": item["question"]["string"],
            "paraphrased_question": item["paraphrased_question"]["string"],
            "query_type": item["query_type"],
            "query": item["query"]["sparql"],
            "entities": item["entities"] 
        })
        
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
