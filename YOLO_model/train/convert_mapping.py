import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil

def convert_json_label(input_file, output_file, verbose=False):
    """
    Convert specific waste category labels to main category labels.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        verbose: Whether to print verbose information
    
    Returns:
        True if conversion was successful, False otherwise
    """
    # Mapping from specific categories to main categories
    category_mapping = {
        # Specific labels -> Main category
        "potato": "Kitchen waste",
        "daikon": "Kitchen waste",
        "carrot": "Kitchen waste",
        "bottle": "Recyclable waste",
        "can": "Recyclable waste",
        "battery": "Hazardous waste",
        "drug": "Hazardous waste",
        "inner_packing": "Hazardous waste",
        "tile": "Other waste",
        "stone": "Other waste",
        "brick": "Other waste",
        # Keep main categories as they are
        "Kitchen waste": "Kitchen waste",
        "Recyclable waste": "Recyclable waste",
        "Hazardous waste": "Hazardous waste",
        "Other waste": "Other waste"
    }
    
    try:
        # Read input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert labels
        if "labels" in data:
            for label in data["labels"]:
                if "name" in label:
                    specific_category = label["name"]
                    if specific_category in category_mapping:
                        main_category = category_mapping[specific_category]
                        if verbose and specific_category != main_category:
                            print(f"Converting '{specific_category}' to '{main_category}' in {input_file}")
                        label["name"] = main_category
                    else:
                        print(f"Warning: Unknown category '{specific_category}' in {input_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write output JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_directory(input_dir, output_dir, num_workers=None, verbose=False):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        num_workers: Number of worker threads to use
        verbose: Whether to print verbose information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in input directory
    input_files = list(Path(input_dir).glob("**/*.json"))
    total_files = len(input_files)
    
    if total_files == 0:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {total_files} JSON files to process")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, min(os.cpu_count(), 8))
    
    print(f"Using {num_workers} worker threads")
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for input_file in input_files:
            # Calculate relative path and construct output path
            rel_path = input_file.relative_to(input_dir)
            output_file = Path(output_dir) / rel_path
            
            # Submit task to executor
            future = executor.submit(convert_json_label, str(input_file), str(output_file), verbose)
            futures.append(future)
        
        # Wait for all tasks to complete
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error in worker: {e}")
                failed += 1
            
            # Print progress
            if (i+1) % 100 == 0 or (i+1) == total_files:
                print(f"Progress: {i+1}/{total_files} files processed")
    
    print(f"\nConversion complete: {successful} successful, {failed} failed")

def process_single_file(input_file, output_file=None, verbose=True):
    """
    Process a single JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, will overwrite input file)
        verbose: Whether to print verbose information
    """
    if output_file is None:
        # Make a backup of the original file
        backup_file = input_file + ".bak"
        shutil.copy2(input_file, backup_file)
        print(f"Backed up original file to {backup_file}")
        output_file = input_file
    
    success = convert_json_label(input_file, output_file, verbose)
    if success:
        print(f"Successfully converted {input_file} to {output_file}")
    else:
        print(f"Failed to convert {input_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert waste labels from specific categories to main categories")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory (if not specified, will overwrite input)")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker threads (default: number of CPU cores)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose information")
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        process_single_file(args.input, args.output, args.verbose)
    elif os.path.isdir(args.input):
        if args.output is None:
            print("Error: Output directory must be specified when input is a directory")
            return
        process_directory(args.input, args.output, args.workers, args.verbose)
    else:
        print(f"Error: Input {args.input} does not exist")

if __name__ == "__main__":
    main()
