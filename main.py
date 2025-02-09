import os
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from config.paths import *
from app.core.load_data import *
from app.core.export_data import *
from app.core.process_data import *

# Load component groups from JSON file
component_groups = load_component_groups(COMPONENT_GROUPS_PATH)

# Export container groups to Excel
export_container_groups_to_excel(component_groups, CONTAINER_GROUPS_OUTPUT_PATH)

# Load data
excluded_words = load_excluded_words(EXCLUDED_WORDS_PATH)
component_groups = load_component_groups(COMPONENT_GROUPS_PATH)
excluded_containers = load_excluded_containers(EXCLUDED_CONTAINERS_PATH)

print(f"Excluded words: {excluded_words}")
print(f"Loaded component groups: {component_groups}")
print(f"Excluded containers: {excluded_containers}")


for file_name in os.listdir(FOLDER_PATH):
    if file_name.endswith('.json'):
        file_path = os.path.join(FOLDER_PATH, file_name)
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
            ngram_range=(1, 7), 
            token_pattern=r"(?u)\b\S+\b",
            lowercase=False
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Save the vocabulary to an Excel file
        vocabulary = vectorizer.vocabulary_
        vocabulary_df = pd.DataFrame(vocabulary.items(), columns=['Term', 'Index'])
        vocabulary_df.to_excel(VOCABULARY_OUTPUT_PATH, index=False)
        print(f"Vocabulary saved to {VOCABULARY_OUTPUT_PATH}")
        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_vec)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # Load the input data from the specified Excel file for prediction
        input_df = pd.read_excel(INPUT_EXCEL_PATH)
        
        # Load JSON data
        with open(COMPONENT_GROUPS_PATH, 'r') as json_file: # Server
        #with open('app/data/component_groups.json', 'r') as json_file: # Local
            json_data = json.load(json_file)
        groups = json_data['Groups']
        
        # Filter rows based on criteria from JSON data
        filtered_rows = input_df[input_df.apply(lambda row: any(row['ComponentGroup'] == group['ComponentGroup'] and row['ContainerName'] in group['ContainerName'] for group in groups), axis=1)]
        rows_to_delete = input_df.index.difference(filtered_rows.index)
        input_df = input_df.drop(rows_to_delete)

        # Drop rows where 'ContainerValue' is '[BLANK]' or null
        input_df = input_df.dropna(subset=['ContainerValue', 'ComponentGroup'])
        input_df = input_df[input_df['ContainerValue'] != '[BLANK]']

        # Exclude rows where 'ContainerName' is in the excluded_containers list
        input_df = input_df[~input_df['ContainerName'].isin(excluded_containers)]

        # Remove trailing semicolons from 'ContainerValue'
        input_df['ContainerValue'] = input_df['ContainerValue'].str.rstrip(';')

        # Ensure 'ContainerValue' column is consistently formatted
        input_df['ContainerValue'] = input_df['ContainerValue'].apply(lambda x: str(x).encode('utf-8').decode('utf-8'))

        # Initialize a dictionary to hold the prediction columns for each file
        prediction_results = {
            'ComponentGroup': input_df['ComponentGroup'].tolist(),
            'ContainerName': input_df['ContainerName'].tolist(),
            'ContainerValue': input_df['ContainerValue'].tolist()
        }

        # Loop through each JSON file to generate predictions separately for each
        for file_name in os.listdir(FOLDER_PATH):
            if file_name.endswith('.json'):
                file_path = os.path.join(FOLDER_PATH, file_name)
                file_label = file_name.replace('.json', '')

                values_for_file = [None] * len(input_df)

                for idx, value in enumerate(input_df['ContainerValue']):
                    words = value.split()

                    matched_container_value = None

                    for word in words:
                        if word in excluded_words:
                            print(f"Word '{word}' is in the excluded list, skipping...")
                            continue

                        if word in vectorizer.vocabulary_:
                            input_vec = vectorizer.transform([word])
                            predicted_tag = model.predict(input_vec)

                            if predicted_tag[0] == file_label:
                                with open(file_path, 'r', encoding='utf-8') as json_file:
                                    try:
                                        data = json.load(json_file)
                                        for entry in data.get(file_label, []):
                                            if 'ContainerValue' in entry:
                                                matched_container_value = entry['ContainerValue']
                                                break
                                    except json.JSONDecodeError as e:
                                        print(f"Error decoding JSON in file {file_path}: {e}")
                                        break

                                break

                    values_for_file[idx] = matched_container_value

                prediction_results[file_label] = values_for_file

        # Convert the prediction results to a DataFrame
        prediction_results_df = pd.DataFrame(prediction_results)

        # Remove columns where all values are NaN
        prediction_results_df = prediction_results_df.dropna(axis=1, how='all')

        # Save the predictions DataFrame to Excel
        prediction_results_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"Granulated predictions saved to {OUTPUT_EXCEL_PATH} with ContainerValues.")

        # Display the final DataFrame structure
        print(prediction_results_df.head())