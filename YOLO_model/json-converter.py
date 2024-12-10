import os
import json
import shutil
from pathlib import Path

def convert_polygon_to_bbox(input_json):
    """
    Convert from polygon format to bounding box format
    """
    output = {"labels": []}
    
    # Process each shape in the input
    for shape in input_json["shapes"]:
        # Get all x coordinates and y coordinates from points
        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Calculate bounding box coordinates
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        
        # Create label entry in required format
        label_entry = {
            "y1": y1,
            "x2": x2,
            "x1": x1,
            "y2": y2,
            "name": shape["label"]
        }
        
        output["labels"].append(label_entry)
    
    return output

def process_folder(input_folder, output_folder):
    """
    Process all JSON files in the input folder and save converted files to output folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear output folder if it exists
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    
    # Get all JSON files in input folder
    json_files = list(Path(input_folder).glob("*.json"))
    
    # Process each JSON file
    for json_file in json_files:
        try:
            # Read input JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                input_json = json.load(f)
            
            # Convert format
            output_json = convert_polygon_to_bbox(input_json)
            
            # Create output file path
            output_file = os.path.join(output_folder, json_file.name)
            
            # Save converted JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2)
            
            print(f"Successfully converted {json_file.name}")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
    
    print(f"\nProcessing complete. Converted {len(json_files)} files.")

def main():
    # Default folders
    default_input = "labels"
    default_output = "trans_labels"
    
    # Get input folder from user
    input_folder = input(f"Enter input folder path (press Enter for default '{default_input}'): ").strip()
    if not input_folder:
        input_folder = default_input
    
    # Get output folder from user
    output_folder = input(f"Enter output folder path (press Enter for default '{default_output}'): ").strip()
    if not output_folder:
        output_folder = default_output
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    # Process files
    print(f"\nProcessing files from '{input_folder}' to '{output_folder}'...")
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
