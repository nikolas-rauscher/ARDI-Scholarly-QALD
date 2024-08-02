import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
FILE_PATH = 'results/human_eval_2_zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
OUTPUT_DIR = 'results/evaluation/nikolas/results'

FEEDBACK_MAPPING = {
    0: 'Correct',
    1: 'Partially Correct',
    2: 'Insufficient Info Correct',
    3: 'Insufficient Info Wrong',
    4: 'Wrong',
    'Token limit': 'Token limit'
}

COLOR_MAPPING = {
    'Insufficient Info Correct': 'orange',
    'Partially Correct': 'blue',
    'Correct': 'green',
    'Wrong': 'red',
    'Insufficient Info Wrong': 'purple',
    'Token limit': 'yellow'
}


def load_json(file_path):
    """Load a JSON file and return the data."""
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def extract_feedback(data):
    """Extract and map feedback values to the given categories."""
    rows = []
    for entry in data:
        for result in entry['results']:
            rows.append({
                'all_triples_feedback': result['all_triples_results']['metrics'].get('human_feedback', 'Token limit'),
                'verbalizer_feedback': result['verbalizer_results']['metrics'].get('human_feedback', 'Token limit'),
                'evidence_matching_feedback': result['evidence_matching']['metrics'].get('human_feedback', 'Token limit'),
                'verbalizer_plus_evidence_feedback': result['verbalizer_plus_evidence_matching']['metrics'].get('human_feedback', 'Token limit')
            })
    return rows


def create_dataframe(feedback_data):
    """Create a DataFrame from the feedback data and replace values with categories."""
    df = pd.DataFrame(feedback_data)
    df.replace(FEEDBACK_MAPPING, inplace=True)
    return df.astype(str)


def plot_feedback(dataframe, filename):
    """Plot the feedback data and save the plot as a PNG file."""
    grouped_data = dataframe.apply(
        pd.Series.value_counts).fillna(0).astype(int).T
    ax = grouped_data.plot(kind='bar', figsize=(14, 8), color=[
                           COLOR_MAPPING.get(x, '#333333') for x in grouped_data.columns])
    plt.xlabel('Feedback Type')
    plt.ylabel('Count')
    plt.title(f'Results for {filename}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Feedback Value', loc='upper right')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename.replace('.json', '.png'))
    plt.savefig(output_path)
    plt.show()


def main():
    filename = os.path.basename(FILE_PATH)
    data = load_json(FILE_PATH)
    feedback_data = extract_feedback(data)
    df_corrected = create_dataframe(feedback_data)
    plot_feedback(df_corrected, filename)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
