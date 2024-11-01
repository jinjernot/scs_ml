import os
import json

folder_path = './app/data/json_ml'

def remove_duplicates(data):
    """
    Remove duplicated 'ContainerValue' entries from lists in the JSON structure.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                # If the value is a list, look for duplicate 'ContainerValue'
                unique_entries = []
                seen_values = set()
                for item in value:
                    if isinstance(item, dict) and "ContainerValue" in item:
                        container_value = item["ContainerValue"]
                        # Only add unique 'ContainerValue' entries
                        if container_value not in seen_values:
                            seen_values.add(container_value)
                            unique_entries.append(item)
                    else:
                        # If not a dict or doesn't have 'ContainerValue', keep the entry
                        unique_entries.append(item)
                # Replace the original list with the filtered unique list
                data[key] = unique_entries
            else:
                # Recursively handle nested dictionaries
                remove_duplicates(value)
    elif isinstance(data, list):
        # If it's a list, recursively check for nested dictionaries
        for item in data:
            remove_duplicates(item)

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # Open and load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Remove duplicate 'ContainerValue' entries
        remove_duplicates(data)
        
        # Write the modified data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

print("Duplicate removal complete!")
