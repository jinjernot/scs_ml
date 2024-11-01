import os
import json

# Path to the folder containing the JSON files
folder_path = './app/data/json_ml'

# Function to keep at least 10 occurrences of each 'ContainerValue' in the JSON data
def keep_minimum_occurrences(json_data, min_occurrences=500):
    # Iterate over all keys in the JSON data
    for key, value in json_data.items():
        if isinstance(value, list):  # Only process if the value is a list
            seen = {}
            unique_entries = []
            
            # Iterate through the list and count occurrences of 'ContainerValue'
            for entry in value:
                container_value = entry['ContainerValue']
                
                if container_value not in seen:
                    seen[container_value] = 0
                
                # If we have less than min_occurrences, append the entry
                if seen[container_value] < min_occurrences:
                    unique_entries.append(entry)
                    seen[container_value] += 1
            
            # Replace the original list with the entries that meet the occurrence criteria
            json_data[key] = unique_entries

    return json_data

# Process each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Keep at least 10 occurrences of each 'ContainerValue' in the JSON
        cleaned_data = keep_minimum_occurrences(data)
        
        # Save the cleaned data back to the file
        with open(file_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)

print("All JSON files have been processed to keep at least 10 occurrences.")
