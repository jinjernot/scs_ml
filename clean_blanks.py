import os
import json
import pandas as pd

# Define the folder path
folder_path = './app/data/json_ml'

# List to store removed records
removed_records = []

# Function to clean up the JSON file
def clean_json_file(file_path):
    global removed_records
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

    cleaned_data = {}
    for key, entries in data.items():
        if isinstance(entries, list):
            cleaned_entries = []
            for entry in entries:
                if isinstance(entry, dict):
                    if entry.get("ContainerValue") == "[BLANK]":
                        removed_records.append({
                            'file': file_path,
                            'key': key,
                            'Component': entry.get('Component'),
                            'ContainerValue': entry.get('ContainerValue')
                        })
                    else:
                        cleaned_entries.append(entry)
                else:
                    print(f"Invalid entry structure in {file_path}: {entry}")
            cleaned_data[key] = cleaned_entries
        else:
            print(f"Expected a list for key '{key}' in {file_path}, but got {type(entries).__name__}")

    # Write the cleaned data back to the file
    with open(file_path, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=4)

# Loop through the files in the folder
for file_name in os.listdir(folder_path):
    # Process only JSON files
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            print(f"Cleaning file: {file_path}")
            clean_json_file(file_path)
        else:
            print(f"File not found: {file_path}")

# Create a DataFrame from the removed records and save it to Excel
if removed_records:
    df_removed = pd.DataFrame(removed_records)
    output_excel_path = './removed_records.xlsx'
    df_removed.to_excel(output_excel_path, index=False)
    print(f"Removed records saved to {output_excel_path}")
else:
    print("No records were removed.")

print("Cleanup completed.")
