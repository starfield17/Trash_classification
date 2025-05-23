import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
from sklearn.model_selection import train_test_split
import cv2
import numpy as np # This import is not used in the provided code, can be removed if not used elsewhere
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained YOLOv model')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to the pre-trained best.pt model')
    parser.add_argument('--data-path', type=str, default='./finetune_data',
                        help='Path to the new data for fine-tuning')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for fine-tuning (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (default: 0.0001)')
    return parser.parse_args()

def validate_json_file(json_path):
    """Validates if a JSON file is valid."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        print(f"Warning: JSON decoding error - invalid JSON file: {json_path}")
        return False
    except UnicodeDecodeError:
        print(f"Warning: Unicode decoding error - invalid JSON file: {json_path}")
        return False
    except Exception as e:
        print(f"Warning: Failed to read JSON file {json_path} - error: {e}")
        return False

def _check_single_file(img_file, data_dir):
    """Checks a single image file and its corresponding JSON label file."""
    img_path = os.path.join(data_dir, img_file)
    base_name = os.path.splitext(img_file)[0]
    json_file = os.path.join(data_dir, base_name + ".json")

    # Check image integrity
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Corrupt or invalid image file: {img_file}")
            return None
        # Check if image dimensions are reasonable
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            print(f"Warning: Image dimensions are too small: {img_file}")
            return None
    except Exception as e:
        print(f"Warning: Error reading image {img_file}: {e}")
        return None

    # Check if the label file exists and is valid
    if os.path.exists(json_file):
        # Ensure the label file actually has a .json extension
        if not json_file.lower().endswith(".json"):
            print(f"Warning: Incorrect label file extension (should be .json): {json_file}")
            return None

        # Validate JSON file content
        if not validate_json_file(json_file):
            return None

        # Validate JSON file structure
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            if "labels" not in label_data:
                print(f"Warning: Invalid label file structure (missing 'labels' key): {json_file}")
                return None
            # If all checks pass, return the image filename
            return img_file
        except Exception as e:
            print(f"Warning: Error processing label file {json_file} (structure check): {e}")
            return None
    else:
        print(f"Warning: Corresponding label file not found: {json_file}")
        return None

def check_and_clean_dataset(data_dir):
    """Checks dataset integrity and cleans invalid data."""
    print("Checking dataset integrity...")
    image_extensions = (".jpg", ".jpeg", ".png")

    try:
        if not os.path.isdir(data_dir):
            print(f"Error: Data directory does not exist or is not a directory: {data_dir}")
            return []
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"Error: Could not list directory {data_dir}: {e}")
        return []

    image_files = [
        f for f in all_files if f.lower().endswith(image_extensions)
    ]
    print(f"Found {len(image_files)} potential image files.")
    if not image_files:
        print("No supported image files found in the directory.")
        return []

    valid_pairs = []
    futures = []
    # Determine maximum number of worker threads
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using up to {max_workers} worker threads for checking...")

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit check tasks for each image file
        for img_file in image_files:
            futures.append(executor.submit(_check_single_file, img_file, data_dir))

        # Collect results
        for future in futures:
            try:
                result = future.result()
                if result:
                    valid_pairs.append(result)
            except Exception as exc:
                print(f'A file check task generated an exception: {exc}')

    print(f"\nCheck complete. Found {len(valid_pairs)} valid image and label file pairs.")
    return valid_pairs

def create_data_yaml():
    """Creates the data configuration file (data.yaml) - uses four main categories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        "path": current_dir,
        "train": os.path.join(current_dir, "train/images"),
        "val": os.path.join(current_dir, "val/images"),
        "test": os.path.join(current_dir, "test/images"),
        "names": {0: "kitchen_waste", 1: "recyclable_waste", 2: "hazardous_waste", 3: "other_waste"},
        "nc": 4,  # 4 major categories
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    return os.path.join(current_dir, "data.yaml")

def try_create_symlink(src, dst):
    """Attempts to create a symbolic link. Returns True on success, False on failure."""
    try:
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception as e:
        print(f"Warning: Could not create symbolic link from {src} to {dst}: {e}")
        return False

def process_split(split_name, files, data_dir, use_symlinks=True):
    """Processes dataset split (image copying/linking and label conversion)."""
    print(f"\nProcessing {split_name} split...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    # Ensure target directories exist
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    # Test if symbolic links can be used
    symlinks_working = False
    if use_symlinks and files:
        test_img = os.path.join(data_dir, files[0])
        test_dst = os.path.join(split_img_dir, "test_symlink_" + files[0])
        symlinks_working = try_create_symlink(test_img, test_dst)

        # Clean up test link
        if os.path.exists(test_dst):
            os.remove(test_dst)

    print(f"{split_name}: Using {'symbolic links' if symlinks_working else 'file copying'} for dataset preparation.")

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file)
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt")

        # Process image and label files
        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                # Handle image files - try symbolic links first if enabled
                if symlinks_working:
                    if not try_create_symlink(src_img, dst_img):
                        # Fallback to copying if symbolic link creation fails
                        shutil.copy2(src_img, dst_img)
                else:
                    # If symbolic links are not available, copy directly
                    shutil.copy2(src_img, dst_img)

                # Convert labels
                convert_labels(src_json, dst_txt)
            except Exception as e:
                print(f"Error processing file pair ({img_file}, {base_name}.json): {e}")
        else:
            if not os.path.exists(src_img):
                print(f"Warning: Source image not found during split processing: {src_img}")
            if not os.path.exists(src_json):
                print(f"Warning: Source JSON not found during split processing: {src_json}")

    print(f"{split_name}: Processed {len(files)} potential images.")

def prepare_dataset(data_dir, valid_pairs, use_symlinks=True):
    """Prepares the dataset - splits into training, validation, and test sets."""
    # Ensure at least 10 images for validation
    if len(valid_pairs) < 10:
        raise ValueError(
            f"Insufficient number of valid data pairs ({len(valid_pairs)}). At least 10 images are required for fine-tuning."
        )

    # Clean existing directories
    print("\nCleaning existing train/val/test directories...")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), split)
        if os.path.exists(split_path):
            print(f"Deleting {split_path} directory...")
            try:
                shutil.rmtree(split_path)
            except OSError as e:
                print(f"Could not directly delete directory {split_path}, error: {e}")
                print(f"Attempting to delete files and subdirectories individually...")

                # Delete files and directories individually
                for root, dirs, files in os.walk(split_path, topdown=False):
                    # Delete files first
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Warning: Could not delete file {file_path}: {e}")

                    # Then delete empty directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            print(f"Warning: Could not delete directory {dir_path}: {e}")

                # Finally, try to delete the main directory
                try:
                    os.rmdir(split_path)
                except Exception as e:
                    print(f"Warning: Could not delete main directory {split_path}: {e}")

        # Recreate directory structure
        print(f"Creating {split} directory structure...")
        for subdir in ["images", "labels"]:
            subdir_path = os.path.join(split_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)

    # For fine-tuning, we use more validation data to ensure model generalization
    # Dataset splitting (70% train, 20% validation, 10% test)
    print("Splitting dataset into training, validation, and test sets...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.33, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Use ThreadPoolExecutor to process each dataset split in parallel
    print("Starting parallel processing of dataset splits...")
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_split, split_name, files, data_dir, use_symlinks): split_name
            for split_name, files in splits.items()
        }

        # Wait for all tasks to complete and check for exceptions
        for future in futures:
            split_name = futures[future]
            try:
                future.result()
                print(f"Finished processing {split_name} split.")
            except Exception as exc:
                print(f'{split_name} split generated an exception: {exc}')

    print("\nDataset preparation complete.")
    print(f"Training set: {len(train_files)} images, Validation set: {len(val_files)} images, Test set: {len(test_files)} images")
    return len(train_files), len(val_files), len(test_files)

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Converts bounding box from x1,y1,x2,y2 to YOLO format."""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # Calculate center coordinates and width/height
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Ensure values are within 0-1 range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height

def convert_labels(json_file, txt_file):
    """Converts labels to the four major categories."""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False

        # Get image path
        base_name = os.path.splitext(json_file)[0]
        possible_extensions = [".jpg", ".jpeg", ".png"]
        img_path = None

        for ext in possible_extensions:
            temp_path = base_name + ext
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            print(f"Warning: Corresponding image file not found for: {json_file}")
            return False

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            return False

        img_height, img_width = img.shape[:2]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Mapping of detailed categories to major categories
        category_mapping = {
            # Kitchen Waste (0)
            "Kitchen_waste": 0,
            "potato": 0,
            "daikon": 0,
            "carrot": 0,
            # Recyclable Waste (1)
            "Recyclable_waste": 1,
            "bottle": 1,
            "can": 1,
            # Hazardous Waste (2)
            "Hazardous_waste": 2,
            "battery": 2,
            "drug": 2,
            "inner_packing": 2,
            # Other Waste (3)
            "Other_waste": 3,
            "tile": 3,
            "stone": 3,
            "brick": 3,
        }

        with open(txt_file, "w", encoding="utf-8") as f:
            if "labels" not in data:
                print(f"Warning: No 'labels' key found in JSON file {json_file}")
                return False

            for label in data["labels"]:
                try:
                    if "name" not in label:
                        print(f"Warning: Label data in {json_file} is missing 'name' field")
                        continue

                    class_name = label["name"]
                    if class_name not in category_mapping:
                        print(f"Warning: Unknown category {class_name} in {json_file}")
                        continue

                    # Use the major category ID directly
                    category_id = category_mapping[class_name]

                    required_keys = ["x1", "y1", "x2", "y2"]
                    if not all(key in label for key in required_keys):
                        print(f"Warning: Missing bounding box coordinates in {json_file}")
                        continue

                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height
                    )

                    if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                        f.write(
                            f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )
                    else:
                        print(f"Warning: Invalid bounding box values in {json_file}")
                        continue
                except KeyError as e:
                    print(f"Warning: Missing key in label data in {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing label in {json_file}: {e}")
                    continue
        return True
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return False

def finetune_yolo(pretrained_model, data_yaml_path, args):
    """
    Fine-tunes a pre-trained YOLOv model.

    Args:
        pretrained_model: Path to the pre-trained model.
        data_yaml_path: Path to the data configuration file.
        args: Command-line arguments.
    """
    print(f"\nLoading pre-trained model: {pretrained_model}")
    try:
        model = YOLO(pretrained_model)
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "0"

    # Adjust batch size for fine-tuning
    if device == "cpu":
        batch_size = min(4, args.batch_size)  # Limit batch size on CPU
        workers = max(1, min(os.cpu_count() - 1, 4))
        use_half = False  # CPU does not support mixed-precision training
    else:
        batch_size = args.batch_size
        workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        use_half = True  # Enable mixed-precision training on GPU for speed
    
    # Configure fine-tuning parameters
    finetune_args = {
        "data": data_yaml_path,
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": batch_size,
        "workers": workers,
        "device": device,
        "patience": args.patience,
        "save_period": 5,
        "exist_ok": True,
        "project": os.path.dirname(os.path.abspath(__file__)),
        "name": "runs/finetune",
        "optimizer": "AdamW",
        "lr0": args.lr,  # Use a smaller learning rate for fine-tuning
        "lrf": 0.01,  # Learning rate decay factor
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,  # Reduce warmup epochs for fine-tuning
        "warmup_momentum": 0.5,
        "warmup_bias_lr": 0.05,
        "box": 7.5,  # Increase bounding box weight for better localization
        "cls": 4.0,  # Increase classification weight
        "dfl": 3.0,
        "nbs": 64,
        "val": True,  # Ensure validation after each epoch
        "rect": True,  # Use rectangular training for efficiency
        "cache": True,
        "single_cls": False,
        "half": use_half,  # Set half precision based on device
        # Data augmentation settings
        "augment": True,
        "degrees": 5.0,  # Reduce rotation angle for fine-tuning
        "scale": 0.1,  # Reduce scaling range for fine-tuning
        "fliplr": 0.5,  # Increase horizontal flip probability
        "hsv_h": 0.015,  # Reduce hue variation
        "hsv_s": 0.1,  # Reduce saturation variation
        "hsv_v": 0.05,  # Reduce brightness variation
        "mosaic": 0.5,  # Use mosaic augmentation
        "mixup": 0.1,  # Keep mixup but reduce intensity
    }

    # Start fine-tuning
    try:
        print(f"\nUsing device: {'GPU' if device == '0' else 'CPU'}")
        print(f"Batch size: {finetune_args['batch']}")
        print(f"Learning rate: {finetune_args['lr0']}")
        print(f"Number of epochs: {finetune_args['epochs']}")
        print(f"Mixed precision training: {'Enabled' if finetune_args['half'] else 'Disabled'}\n")

        results = model.train(**finetune_args)

        # Return fine-tuning results
        print("\nFine-tuning complete!")
        return results
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_finetuned_model(model_dir):
    """
    Saves the fine-tuned model and creates a model information file.

    Args:
        model_dir: Directory where the fine-tuned model is saved.
    """
    best_pt_path = os.path.join(model_dir, 'weights', 'best.pt')

    if not os.path.exists(best_pt_path):
        print(f"Error: Fine-tuned model {best_pt_path} not found.")
        return

    # Create an info file to record fine-tuning information
    info_path = os.path.join(model_dir, 'finetuned_model_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("Fine-tuned Model Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Path: {best_pt_path}\n")
        f.write(f"Fine-tuning Time: {Path(best_pt_path).stat().st_mtime}\n")
        f.write(f"Model Size: {Path(best_pt_path).stat().st_size / (1024*1024):.2f} MB\n")
        f.write("Fine-tuned Model Configuration:\n")
        f.write("- Fine-tuned using pre-trained weights\n")
        f.write("- Applicable for trash classification (Kitchen waste, Recyclable waste, Hazardous waste, Other waste)\n")

    print(f"\nFine-tuned model saved to: {best_pt_path}")
    print(f"Model information saved to: {info_path}")

    # Copy a versioned backup
    version = 1
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuned_models')
    os.makedirs(backup_dir, exist_ok=True)

    # Find the current highest version
    existing_models = list(Path(backup_dir).glob('finetuned_v*.pt'))
    if existing_models:
        versions = [int(m.stem.split('_v')[1]) for m in existing_models]
        version = max(versions) + 1

    backup_path = os.path.join(backup_dir, f'finetuned_v{version}.pt')
    shutil.copy2(best_pt_path, backup_path)
    print(f"Backup model saved to: {backup_path}")

def main():
    """Main function - fine-tunes a pre-trained YOLOv model."""
    # Parse command-line arguments
    args = parse_args()

    try:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check if pre-trained model exists
        if not os.path.exists(args.pretrained):
            print(f"Error: Pre-trained model {args.pretrained} does not exist.")
            return

        data_dir = args.data_path

        # 1. Check dataset
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(data_dir)
        if not valid_pairs:
            print("No valid data pairs found. Exiting.")
            return
        gc.collect()

        # 2. Create data.yaml
        print("\nStep 2: Creating data.yaml...")
        data_yaml_path = create_data_yaml()

        # 3. Prepare dataset
        print("\nStep 3: Preparing dataset...")
        try:
            train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs, use_symlinks=True)
            gc.collect()
            if val_size < 3:
                print(f"Warning: Validation set size ({val_size}) is less than 3. Fine-tuning might not yield good results.")
        except ValueError as ve:
            print(f"Error during dataset preparation: {ve}")
            return

        # 4. Fine-tune model
        print("\nStep 4: Starting fine-tuning...")
        results = finetune_yolo(args.pretrained, data_yaml_path, args)

        # 5. Save fine-tuned model
        if results:
            print("\nStep 5: Saving fine-tuned model...")
            # If results object has a save_dir attribute, use it as the model directory
            if hasattr(results, 'save_dir'):
                model_dir = results.save_dir
            else:
                # Otherwise, construct the expected path
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/finetune')

            save_finetuned_model(model_dir)
        else:
            print("\nFine-tuning did not complete successfully or was interrupted. Skipping model saving.")

    except Exception as e:
        print(f"\nAn error occurred in the main execution flow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript finished. Cleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
