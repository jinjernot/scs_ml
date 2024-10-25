import os
import json

# Path to the folder containing the JSON files
folder_path = './app/data/json_ml'

# Function to remove duplicates from lists in a JSON file based on 'ContainerValue'
def remove_duplicates_from_lists(json_data):
    # Iterate over all keys in the JSON data
    for key, value in json_data.items():
        if isinstance(value, list):  # Only process if the value is a list
            seen = set()
            unique_entries = []
            
            # Iterate through the list and check for duplicates in 'ContainerValue'
            for entry in value:
                if entry['ContainerValue'] not in seen:
                    unique_entries.append(entry)
                    seen.add(entry['ContainerValue'])
            
            # Replace the original list with the unique entries
            json_data[key] = unique_entries

    return json_data

# Process each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Remove duplicates from all lists in the JSON
        cleaned_data = remove_duplicates_from_lists(data)
        
        # Save the cleaned data back to the file
        with open(file_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)

print("All JSON files have been cleaned up.")
