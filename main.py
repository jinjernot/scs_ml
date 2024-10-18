import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Define the folder path for JSON files and the Excel file path for predictions
folder_path = './app/data/json_ml'
input_excel_path = './file.xlsx'  # Path to your input Excel file with the "ContainerValue" column
output_excel_path = './granulated_predictions.xlsx'  # Path to save the new Excel file with predictions

# List to store records from all files
records = []

# Function to process JSON file and collect ContainerValue and file label
def process_json_file(file_path, file_label):
    global records
    # Read the JSON content
    with open(file_path, 'r') as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}")
            return

    # Check that the data is a dictionary
    if not isinstance(data, dict):
        print(f"Unexpected structure in {file_path}, skipping...")
        return

    # Extract ContainerValue and create records
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

# Loop through the files in the folder
for file_name in os.listdir(folder_path):
    # Process only JSON files
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            # Use file name (without extension) as the label for the file
            file_label = file_name.replace('.json', '')
            process_json_file(file_path, file_label)
        else:
            print(f"File not found: {file_path}")

# Create a DataFrame from the collected records
df = pd.DataFrame(records)

# Check if the DataFrame is empty
if df.empty:
    print("The DataFrame is empty. Please check your data extraction process.")
else:
    print("DataFrame created successfully.")
    print(df.head())  # Display the first few rows to verify the content

    # Drop rows with missing values in 'ContainerValue' or 'FileLabel'
    df = df.dropna(subset=['ContainerValue', 'FileLabel'])

    if df.empty:
        print("The DataFrame is empty after removing rows with missing values.")
    else:
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            df['ContainerValue'], df['FileLabel'], test_size=0.2, random_state=42
        )

        # Vectorize the 'ContainerValue' text using TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train a Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_vec)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # Load the input data from the specified Excel file for prediction
        input_df = pd.read_excel(input_excel_path)  # Load the Excel file
        if 'ContainerValue' not in input_df.columns:
            print(f"'ContainerValue' column not found in {input_excel_path}.")
        else:
            # Get the ContainerValue data for prediction, ignoring '[BLANK]'
            input_values = input_df['ContainerValue'].dropna().tolist()
            input_values = [value for value in input_values if value != '[BLANK]']  # Ignore '[BLANK]'

            # Create a DataFrame to hold predictions for each input value
            input_predictions = []

            for value in input_values:
                # Split the value into individual words
                words = value.split()  # Split by spaces
                
                for word in words:
                    # Vectorize the word
                    input_vec = vectorizer.transform([word])  
                    predicted_tag = model.predict(input_vec)  # Make the prediction
                    
                    # Append the prediction result
                    input_predictions.append({
                        'ContainerValue': word,
                        'PredictedTag': predicted_tag[0]
                    })

            # Convert predictions to DataFrame
            input_predictions_df = pd.DataFrame(input_predictions)

            # Save the input predictions to an Excel file
            input_predictions_df.to_excel(output_excel_path, index=False)
            print(f"Granulated predictions saved to {output_excel_path}")

            # Print the predicted tags for the input values
            for prediction in input_predictions:
                print(f"ContainerValue: '{prediction['ContainerValue']}' - Predicted Tag: {prediction['PredictedTag']}")
