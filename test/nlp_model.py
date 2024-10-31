import os
import json
import spacy
import random
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


print("Loading SpaCy model...")
nlp = spacy.blank("en")


folder_path = './app/data/json_ml'
input_excel_path = './file.xlsx'
output_excel_path = './granulated_predictions.xlsx'
excluded_words_path = './app/data/excluded_words.json'
component_groups_path = './app/data/component_groups.json'
excluded_containers_path = './app/data/excluded_containers.json'
vocabulary_output_path = './vocabulary.xlsx'
model_output_dir = './spacy_textcat_model'

records = []

def load_excluded_words(file_path):
    print(f"Loading excluded words from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("excluded_words", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading excluded words: {e}")
        return []

# Load component groups
def load_component_groups(file_path):
    print(f"Loading component groups from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("component_groups", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading component groups: {e}")
        return []

# Load excluded containers
def load_excluded_containers(file_path):
    print(f"Loading excluded containers from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("excluded_containers", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading excluded containers: {e}")
        return []

excluded_words = load_excluded_words(excluded_words_path)
component_groups = load_component_groups(component_groups_path)
excluded_containers = load_excluded_containers(excluded_containers_path)

# Process JSON files
def process_json_file(file_path, file_label):
    global records
    print(f"Processing JSON file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}")
            return
    if not isinstance(data, dict):
        print(f"Unexpected structure in {file_path}, skipping...")
        return

    for key, entries in data.items():
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and 'ContainerValue' in entry:
                    records.append({
                        'FileLabel': file_label,
                        'ContainerValue': entry['ContainerValue']
                    })

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        file_label = file_name.replace('.json', '')
        process_json_file(file_path, file_label)

df = pd.DataFrame(records)

if df.empty:
    print("The DataFrame is empty. Please check your data extraction process.")
else:
    print("DataFrame loaded successfully.")
    df = df.dropna(subset=['ContainerValue', 'FileLabel'])

    if df.empty:
        print("The DataFrame is empty after removing rows with missing values.")
    else:
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['ContainerValue'], df['FileLabel'], test_size=0.2, random_state=42
        )

        print("Preparing DocBin for SpaCy training...")
        doc_bin_train = DocBin()
        doc_bin_test = DocBin()

        train_examples = []
        test_examples = []

        for text, label in zip(X_train, y_train):
            doc = nlp.make_doc(text)
            doc.cats = {label: 1.0}
            train_examples.append(Example.from_dict(doc, {'cats': doc.cats}))

        for text, label in zip(X_test, y_test):
            doc = nlp.make_doc(text)
            doc.cats = {label: 1.0}
            test_examples.append(Example.from_dict(doc, {'cats': doc.cats}))

        print("Saving training and test data in 'spacy_data' folder...")
        Path("spacy_data").mkdir(parents=True, exist_ok=True)
        doc_bin_train.to_disk("spacy_data/train.spacy")
        doc_bin_test.to_disk("spacy_data/test.spacy")

        if 'textcat' not in nlp.pipe_names:
            print("Adding text categorizer to pipeline...")
            textcat = nlp.add_pipe("textcat")
            for label in df['FileLabel'].unique():
                textcat.add_label(label)

        print("Initializing SpaCy model...")
        nlp.initialize(lambda: iter(train_examples))

        print("Starting training...")
        optimizer = nlp.resume_training()
        for epoch in range(10):
            losses = {}
            batches = minibatch(train_examples, size=8)
            for batch in batches:
                nlp.update(batch, losses=losses, drop=0.5, sgd=optimizer)
            print(f"Epoch {epoch + 1}, Loss: {losses}")

        print(f"Training complete. Saving model to {model_output_dir}...")
        nlp.to_disk(model_output_dir)
        print(f"Model saved to {model_output_dir}")

        print("Loading input Excel file for predictions...")
        input_df = pd.read_excel(input_excel_path)
        input_df = input_df.dropna(subset=['ContainerValue', 'ComponentGroup'])
        input_df = input_df[input_df['ContainerValue'] != '[BLANK]']
        input_df = input_df[~input_df['ContainerName'].isin(excluded_containers)]
        input_df['ContainerValue'] = input_df['ContainerValue'].str.rstrip(';')

        print("Making predictions...")
        predictions = []
        for text in input_df['ContainerValue']:
            doc = nlp(text)
            predictions.append(max(doc.cats, key=doc.cats.get))

        input_df['PredictedLabel'] = predictions

        print(f"Saving predictions to {output_excel_path}...")
        input_df.to_excel(output_excel_path, index=False)
        print(f"Predictions saved to {output_excel_path}")