import os
import json

# Path to the folder containing JSON files
folder_path = './app/data/json_ml'

# Function to delete JSON files with fewer than 4 pairs of data
def delete_small_json_files(folder_path, min_pairs=4):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    
                    # Ensure the data is a dictionary or a list
                    if isinstance(data, dict):
                        num_pairs = sum(len(v) if isinstance(v, list) else 0 for v in data.values())
                    elif isinstance(data, list):
                        num_pairs = len(data)
                    else:
                        print(f"Unexpected structure in {file_path}, skipping...")
                        continue
                    
                    # Delete the file if it has fewer than min_pairs of data
                    if num_pairs < min_pairs:
                        print(f"Deleting {file_path} - Only {num_pairs} pairs found")
                        os.remove(file_path)
                    else:
                        print(f"{file_path} has {num_pairs} pairs, keeping it.")

            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing {file_path}: {e}")

# Call the function to delete files
delete_small_json_files(folder_path)
