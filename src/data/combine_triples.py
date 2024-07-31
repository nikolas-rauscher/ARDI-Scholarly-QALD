import json
import pandas as pd


def merge_triples(all_triples):
    merged_triples = []

    for entry in all_triples:
        triples = entry.get('triples', [])
        merged_triples.extend(triples)

    return merged_triples


def create_merged_dataframe(df):
    # Construct records with merged triples and additional fields
    records = []
    for _, row in df.iterrows():
        merged_triples = merge_triples(row.get("all_triples", []))
        record = {
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "author_dblp_uri": row.get("author_dblp_uri", ""),
            "triples_number": row.get("triples_number", 0),
            "all_triples": merged_triples
        }
        records.append(record)

    # Convert records to a DataFrame
    return pd.DataFrame(records)


def merge_triples_and_create_dataset(alex_df, dblp_df):
    # Merge the 'all_triples' column, which is a list, from both dataframes
    alex_df['all_triples'] = alex_df['all_triples'].apply(
        lambda x: x if isinstance(x, list) else [])
    dblp_df['all_triples'] = dblp_df['all_triples'].apply(
        lambda x: x if isinstance(x, list) else [])

    result_df = alex_df.merge(dblp_df, on='id', suffixes=('_alex', '_dblp'))
    result_df['all_triples'] = result_df['all_triples_alex'] + \
        result_df['all_triples_dblp']

    # Create a new dataframe with the required columns
    merged_df = pd.DataFrame({
        "id": alex_df["id"],
        "question": alex_df["question"],
        "author_dblp_uri": dblp_df["author_dblp_uri"],
        "triples_number": result_df['all_triples'].apply(len),
        "all_triples": result_df['all_triples']
    })

    return merged_df


if __name__ == "__main__":
    # Load the datasets
    test_alex_df = pd.read_json(
        "./data/processed/test/ALEX/post_test_data_alex.json")
    test_dblp_df = pd.read_json(
        "./data/processed/test/DBLP/post_test_data_dblp.json")
    train_alex_df = pd.read_json(
        "./data/processed/train/ALEX/openalex_train.json")
    train_dblp_df = pd.read_json(
        "./data/processed/train/DBLP/DBLP_traindata.json")

    # Create merged DataFrames
    merged_test_alex_df = create_merged_dataframe(test_alex_df)
    merged_test_dblp_df = create_merged_dataframe(test_dblp_df)
    merged_train_alex_df = create_merged_dataframe(train_alex_df)
    merged_train_dblp_df = create_merged_dataframe(train_dblp_df)

    # Merge and create the test and train datasets
    test_merged_df = merge_triples_and_create_dataset(
        merged_test_alex_df, merged_test_dblp_df)
    train_merged_df = merge_triples_and_create_dataset(
        merged_train_alex_df, merged_train_dblp_df)

    # Save the merged datasets to JSON files
    test_merged_df.to_json("./data/processed/test.json",
                           orient='records', indent=2)
    train_merged_df.to_json("./data/processed/train.json",
                            orient='records', indent=2)
