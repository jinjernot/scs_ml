import os
import json
import pandas as pd
import numpy as np
import re  # Import regex library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
folder_path = './app/data/json_ml'
input_excel_path = './file.xlsx'
output_excel_path = './granulated_predictions_lstm.xlsx'
excluded_words_path = './app/data/excluded_words.json'
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
        # Tokenize the text data
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['ContainerValue'])

        X = tokenizer.texts_to_sequences(df['ContainerValue'])
        X = pad_sequences(X, maxlen=100)

        # Convert labels to numerical format
        y = pd.get_dummies(df['FileLabel']).values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
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

        # Tokenize and pad the input data for prediction
        input_sequences = tokenizer.texts_to_sequences(input_df['ContainerValue'])
        input_padded = pad_sequences(input_sequences, maxlen=100)

        # Make predictions
        predictions = model.predict(input_padded)

        # Get the predicted labels
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [list(df['FileLabel'].unique())[i] for i in predicted_labels]

        # Add predictions to the DataFrame
        input_df['PredictedLabel'] = predicted_labels

        # Save the predictions DataFrame to Excel
        input_df.to_excel(output_excel_path, index=False)
        print(f"Granulated predictions saved to {output_excel_path}.")

        # Display the final DataFrame structure
        print(input_df.head())
