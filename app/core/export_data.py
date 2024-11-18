import pandas as pd

# Load container names within each component group and export to Excel
def export_container_groups_to_excel(component_groups, output_path):
    # Check if component_groups has data
    if not component_groups:
        print("No component groups loaded. Please check the JSON structure and path.")
        return

    # Create a list to hold data for export
    container_group_data = []
    
    # Iterate through each group in component_groups
    for group in component_groups:
        # Get the component group name
        group_name = group.get("ComponentGroup", "Unknown Group")
        
        # Get the list of container names
        containers = group.get("ContainerName", [])
        
        # Check if there are containers in the current group
        if not containers:
            print(f"No containers found in group '{group_name}'")
            continue

        # Add each container name along with its group to the export list
        for container_name in containers:
            container_group_data.append({
                "GroupName": group_name,
                "ContainerName": container_name
            })

    # Convert the data to a DataFrame
    container_groups_df = pd.DataFrame(container_group_data)
    
    # Check if the DataFrame is not empty before exporting
    if not container_groups_df.empty:
        container_groups_df.to_excel(output_path, index=False)
        print(f"Container groups exported to {output_path}")
    else:
        print("No container data to export.")