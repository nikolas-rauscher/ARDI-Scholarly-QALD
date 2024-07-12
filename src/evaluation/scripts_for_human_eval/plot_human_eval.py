import json
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'human_eval_2_zero_shot_prompting_Mistral-7B-Instruct-v0.2.json'
filename = os.path.basename(file_path)

# Load the JSON file
with open(file_path) as f:
    data = json.load(f)


# Extract relevant information and map feedback values to the given categories
rows = []
for entry in data:
    for result in entry['results']:
        # Check for token limit exceeded case and map feedback values
        all_triples_feedback = 'Token limit' if 'human_feedback' not in result['all_triples_results']['metrics'] else result['all_triples_results']['metrics']['human_feedback']
        verbalizer_feedback = 'Token limit' if 'human_feedback' not in result['verbalizer_results']['metrics'] else result['verbalizer_results']['metrics']['human_feedback']
        evidence_matching_feedback = 'Token limit' if 'human_feedback' not in result['evidence_matching']['metrics'] else result['evidence_matching']['metrics']['human_feedback']
        verbalizer_plus_evidence_feedback = 'Token limit' if 'human_feedback' not in result['verbalizer_plus_evidence_matching']['metrics'] else result['verbalizer_plus_evidence_matching']['metrics']['human_feedback']
        
        rows.append({
            'all_triples_feedback': all_triples_feedback,
            'verbalizer_feedback': verbalizer_feedback,
            'evidence_matching_feedback': evidence_matching_feedback,
            'verbalizer_plus_evidence_feedback': verbalizer_plus_evidence_feedback
        })

# Create a DataFrame
df_corrected = pd.DataFrame(rows)

# Replace feedback values with corresponding categories
feedback_mapping = {
    0: 'Correct',
    1: 'Partially Correct',
    2: 'Insufficient Info Correct',
    3: 'Insufficient Info Wrong',
    4: 'Wrong',
    'Token limit': 'Token limit'
}

df_corrected.replace(feedback_mapping, inplace=True)

# Convert all values to strings to avoid type errors
df_corrected = df_corrected.astype(str)

# Grouping by feedback type and counting the occurrences
grouped_data_corrected = df_corrected.apply(pd.Series.value_counts).fillna(0).astype(int)

# Ensure proper index alignment
grouped_data_corrected = grouped_data_corrected.T

# Plotting the grouped data
color_mapping = {
   'Insufficient Info Correct': 'orange',
    'Partially Correct': 'blue',
    'Correct': 'green',
    'Wrong': 'red',
    'Insufficient Info Wrong': 'purple',
    'Token limit': 'yellow'
}

ax = grouped_data_corrected.plot(kind='bar', figsize=(14, 8), color=[color_mapping.get(x, '#333333') for x in grouped_data_corrected.columns])
plt.xlabel('Feedback Type')
plt.ylabel('Count')
plt.title(f'Results for {filename}') 
plt.xticks(rotation=45, ha='right')
plt.legend(title='Feedback Value', loc='upper right')
plt.tight_layout()

# Save the plot as a PNG file
output_dir = 'src/evaluation/nikolas/results'  
output_path = os.path.join(output_dir, filename.replace('.json', '.png'))
plt.savefig(output_path)

# Display the plot
plt.show()