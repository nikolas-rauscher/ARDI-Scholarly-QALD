from datasets import load_dataset
import json

# Lade den DBLP-QuAD Datensatz
ds = load_dataset("awalesushil/DBLP-QuAD")

# Liste für die neuen Daten
new_data = []

# Iteriere über die Datensätze
for item in ds['train']:
    new_data.append({
        "id": item["id"],
        "question": item["question"]["string"],
        "paraphrased_question": item["paraphrased_question"]["string"],
        "query_type": item["query_type"],
        "query": item["query"]["sparql"],
        "entities": item["entities"]  # Nur die Entitäten extrahieren
    })

# Speichere die neue Liste in einer JSON-Datei
with open('my_questions_and_queries_with_entities.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print("Datensatz erfolgreich erstellt!")