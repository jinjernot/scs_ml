import os
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Paths
folder_path = './app/data/json_ml'
input_excel_path = './file.xlsx'  
output_excel_path = './granulated_predictions.xlsx'  
excluded_words_path = './app/data/excluded_words.json'
component_groups_path = './app/data/component_groups.json'
excluded_containers_path = './app/data/excluded_containers.json'
vocabulary_output_path = './vocabulary.xlsx'

records = []

# Load excluded words
def load_excluded_words(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("excluded_words", [])
    except FileNotFoundError:
        print(f"Excluded words file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {file_path}")
        return []

# Load component groups
def load_component_groups(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("component_groups", [])
    except FileNotFoundError:
        print(f"Component groups file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {file_path}")
        return []

# Load excluded containers
def load_excluded_containers(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data.get("excluded_containers", [])
    except FileNotFoundError:
        print(f"Excluded containers file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {file_path}")
        return []

# Load data
excluded_words = load_excluded_words(excluded_words_path)
component_groups = load_component_groups(component_groups_path)
excluded_containers = load_excluded_containers(excluded_containers_path)

print(f"Excluded words: {excluded_words}")
print(f"Loaded component groups: {component_groups}")
print(f"Excluded containers: {excluded_containers}")

# Process JSON files to include data from all files for training
def process_json_file(file_path, file_label):
    global records
    
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
                else:
                    print(f"Invalid entry structure in {file_path}: {entry}")
        else:
            print(f"Expected a list for key '{key}' in {file_path}, but got {type(entries).__name__}")

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            file_label = file_name.replace('.json', '')
            process_json_file(file_path, file_label)
        else:
            print(f"File not found: {file_path}")

df = pd.DataFrame(records)

if df.empty:
    print("The DataFrame is empty. Please check your data extraction process.")
else:
    print("DataFrame created successfully.")
    print(df.head())  

    df = df.dropna(subset=['ContainerValue', 'FileLabel'])

    if df.empty:
        print("The DataFrame is empty after removing rows with missing values.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['ContainerValue'], df['FileLabel'], test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), 
            token_pattern=r"(?u)\b\S+\b",
            lowercase=False
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Save the vocabulary to an Excel file
        vocabulary = vectorizer.vocabulary_
        vocabulary_df = pd.DataFrame(vocabulary.items(), columns=['Term', 'Index'])
        vocabulary_df.to_excel(vocabulary_output_path, index=False)
        print(f"Vocabulary saved to {vocabulary_output_path}")
        
        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Log predictions on the training data
        y_train_pred = model.predict(X_train_vec)
        train_predictions_df = pd.DataFrame({
            'Actual': y_train,
            'Predicted': y_train_pred
        })
        amd_train_predictions = train_predictions_df[train_predictions_df['Predicted'] == 'AMD']
        print("Training predictions where label is 'AMD':")
        print(amd_train_predictions)

        # Save train predictions to CSV
        train_predictions_df.to_csv("train_predictions_log.csv", index=False)
        print("Training predictions saved to train_predictions_log.csv.")

        # Make predictions on the test set
        y_test_pred = model.predict(X_test_vec)

        # Log test predictions
        test_predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred
        })
        amd_test_predictions = test_predictions_df[test_predictions_df['Predicted'] == 'AMD']
        print("Test predictions where label is 'AMD':")
        print(amd_test_predictions)

        # Save test predictions to CSV
        test_predictions_df.to_csv("test_predictions_log.csv", index=False)
        print("Test predictions saved to test_predictions_log.csv.")

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
