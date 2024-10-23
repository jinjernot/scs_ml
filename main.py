import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Paths
folder_path =           './app/data/json_ml'
input_excel_path =      './file.xlsx'  
output_excel_path =     './granulated_predictions.xlsx'  
excluded_words_path =   './app/data/excluded_words.json'
component_groups_path = './app/data/component_groups.json'
excluded_containers_path = './app/data/excluded_containers.json'

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

# Process JSON files
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
                        'FileLabel': file_label,  # This will be used as the target label
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

        # Vectorize the text data
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_vec)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # Load the input data from the specified Excel file for prediction
        input_df = pd.read_excel(input_excel_path)  # Load the Excel file

        # Drop rows where 'ContainerValue' is '[BLANK]' or null
        input_df = input_df.dropna(subset=['ContainerValue', 'ComponentGroup'])  # Include 'ComponentGroup' in the drop condition
        input_df = input_df[input_df['ContainerValue'] != '[BLANK]']  # Drop rows where 'ContainerValue' is '[BLANK]'

        # Exclude rows where 'ContainerName' is in the excluded_containers list
        input_df = input_df[~input_df['ContainerName'].isin(excluded_containers)]  # Exclude rows based on 'ContainerName'

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

        # Get the number of rows in the input DataFrame
        num_rows = len(input_df)

        # Loop through each file to generate predictions separately for each
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                file_label = file_name.replace('.json', '')

                # Create a list filled with None initially, so it has the same length as 'ContainerValue'
                predictions_for_file = [None] * num_rows

                for idx, value in enumerate(input_df['ContainerValue']):
                    # Split the value into individual words
                    words = value.split()  # Split based on whitespace

                    predicted_word = None

                    # Try to predict based on each word in the ContainerValue
                    for word in words:
                        if word in excluded_words:
                            print(f"Word '{word}' is in the excluded list, skipping...")
                            continue

                        if word in vectorizer.vocabulary_:
                            # Vectorize the word
                            input_vec = vectorizer.transform([word])  # Vectorize each word individually
                            predicted_tag = model.predict(input_vec)

                            # Only set the word if the prediction matches the current file label
                            if predicted_tag[0] == file_label:
                                predicted_word = word  # Save the actual word, not the tag
                                break  # Stop after the first match

                    # Add the predicted word (or None if no valid prediction found)
                    predictions_for_file[idx] = predicted_word

                # Add the predictions to the results dictionary using the file label as the column name
                prediction_results[file_label] = predictions_for_file

        # Convert the prediction results to a DataFrame
        prediction_results_df = pd.DataFrame(prediction_results)

        # Save the predictions DataFrame to Excel
        prediction_results_df.to_excel(output_excel_path, index=False)
        print(f"Granulated predictions saved to {output_excel_path}")

        # Display the final DataFrame structure
        print(prediction_results_df.head())
