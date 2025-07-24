"""
Data utilities module for YOLO training application.
Contains all dataset processing and validation functions.
"""

import os
import json
import yaml
import cv2
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from config import CATEGORY_MAPPING, DATA_CONFIG, IMAGE_EXTENSIONS


def validate_json_file(json_path):
    """
    Validates if a JSON file is valid.
    Returns True if valid, False otherwise.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        print(f"Warning: JSON decode error - invalid JSON file: {json_path}")
        return False
    except UnicodeDecodeError:
        print(f"Warning: Unicode decode error - invalid JSON file: {json_path}")
        return False
    except Exception as e:
        print(f"Warning: Cannot read JSON file {json_path} - error: {e}")
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
            print(f"Warning: Corrupted or invalid image file: {img_file}")
            return None
        # Check if image dimensions are reasonable
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            print(f"Warning: Image dimensions too small: {img_file}")
            return None
    except Exception as e:
        print(f"Warning: Error reading image {img_file}: {e}")
        return None

    # Check if label file exists and is valid
    if os.path.exists(json_file):
        # Ensure label file is indeed a JSON file
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
            return img_file
        except Exception as e:
            print(f"Warning: Error processing label file {json_file} (structure check): {e}")
            return None
    else:
        print(f"Warning: Corresponding label file not found: {json_file}")
        return None


def check_and_clean_dataset(data_dir):
    """Checks dataset integrity and cleans invalid data (Parallelized Version)"""
    print("Checking dataset integrity (parallel)...")
    
    try:
        if not os.path.isdir(data_dir):
            print(f"Error: Data directory does not exist or is not a directory: {data_dir}")
            return []
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"Error: Cannot list directory {data_dir}: {e}")
        return []

    image_files = [
        f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    print(f"Found {len(image_files)} potential image files")
    if not image_files:
        print("No supported image files found in the directory.")
        return []

    valid_pairs = []
    futures = []
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using up to {max_workers} worker threads for checking...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img_file in image_files:
            futures.append(executor.submit(_check_single_file, img_file, data_dir))

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
    """Create data configuration file - using four major categories"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        "path": current_dir,
        "train": os.path.join(current_dir, "train/images"),
        "val": os.path.join(current_dir, "val/images"),
        "test": os.path.join(current_dir, "test/images"),
        "names": DATA_CONFIG["names"],
        "nc": DATA_CONFIG["nc"],
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def try_create_symlink(src, dst):
    """Try to create a symbolic link and return True if successful, False otherwise."""
    try:
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception as e:
        print(f"Warning: Could not create symbolic link from {src} to {dst}: {e}")
        return False


def process_split(split_name, files, data_dir, use_symlinks=True):
    """
    Processes image copying/linking and label conversion for a given dataset split.
    """
    print(f"\nProcessing {split_name} split...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)
    
    symlinks_working = False
    if use_symlinks and files:
        test_img = os.path.join(data_dir, files[0])
        test_dst = os.path.join(split_img_dir, "test_symlink_" + files[0])
        symlinks_working = try_create_symlink(test_img, test_dst)
        
        if os.path.exists(test_dst):
            os.remove(test_dst)
    
    print(f"{split_name}: Using {'symbolic links' if symlinks_working else 'file copying'} for dataset preparation")

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file)
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt")

        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                if symlinks_working:
                    if not try_create_symlink(src_img, dst_img):
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
                
                convert_labels(src_json, dst_txt)
            except Exception as e:
                print(f"Error processing file pair ({img_file}, {base_name}.json): {e}")
        else:
            if not os.path.exists(src_img):
                print(f"Warning: Source image not found during split processing: {src_img}")
            if not os.path.exists(src_json):
                print(f"Warning: Source JSON not found during split processing: {src_json}")

    print(f"{split_name}: Processed {len(files)} potential images")


def prepare_dataset(data_dir, valid_pairs, use_symlinks=True):
    """Prepare dataset - supports using symbolic links instead of copying files to save space"""
    if len(valid_pairs) < 15:
        raise ValueError(
            f"Insufficient number of valid data pairs ({len(valid_pairs)}). At least 15 images are required."
        )
    
    print("\nCleaning existing train/val/test directories...")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), split)
        if os.path.exists(split_path):
            print(f"Deleting {split_path} directory...")
            try:
                shutil.rmtree(split_path)
            except OSError as e:
                print(f"Cannot delete directory {split_path} directly, error: {e}")
                print(f"Attempting to delete files and subdirectories individually...")
                
                for root, dirs, files in os.walk(split_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Warning: Cannot delete file {file_path}: {e}")
                    
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                            print(f"Deleted directory: {dir_path}")
                        except Exception as e:
                            print(f"Warning: Cannot delete directory {dir_path}: {e}")
                
                try:
                    os.rmdir(split_path)
                    print(f"Successfully deleted {split_path}")
                except Exception as e:
                    print(f"Warning: Cannot delete main directory {split_path}: {e}")
                    print(f"Will attempt to continue processing...")
        
        print(f"Creating {split} directory structure...")
        for subdir in ["images", "labels"]:
            subdir_path = os.path.join(split_path, subdir)
            try:
                os.makedirs(subdir_path, exist_ok=True)
                print(f"Created directory: {subdir_path}")
            except Exception as e:
                print(f"Error: Cannot create directory {subdir_path}: {e}")
                raise

    # Dataset splitting (90% train, 5% validation, 5% test)
    print("Splitting dataset into training, validation, and test sets...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.1, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    print("Starting parallel processing of dataset splits...")
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_split, split_name, files, data_dir, use_symlinks): split_name
            for split_name, files in splits.items()
        }

        for future in futures:
            split_name = futures[future]
            try:
                future.result()
                print(f"Finished processing {split_name} split.")
            except Exception as exc:
                print(f'{split_name} split generated an exception: {exc}')
                print(f'Attempting to continue processing other splits...')

    print("\nDataset preparation complete.")
    print(f"Training set: {len(train_files)} images, Validation set: {len(val_files)} images, Test set: {len(test_files)} images")
    return len(train_files), len(val_files), len(test_files)


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box from x1,y1,x2,y2 to YOLO format"""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_labels(json_file, txt_file):
    """Convert to four major categories"""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False
        
        base_name = os.path.splitext(json_file)[0]
        possible_extensions = [".jpg", ".jpeg", ".png"]
        img_path = None

        for ext in possible_extensions:
            temp_path = base_name + ext
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None:
            print(f"Warning: No corresponding image file found for: {json_file}")
            return False
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image: {img_path}")
            return False
        img_height, img_width = img.shape[:2]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(txt_file, "w", encoding="utf-8") as f:
            if "labels" not in data:
                print(f"Warning: No 'labels' key in {json_file}")
                return False
                
            for label in data["labels"]:
                try:
                    if "name" not in label:
                        print(f"Warning: No 'name' field in label data in {json_file}")
                        continue
                        
                    class_name = label["name"]
                    if class_name not in CATEGORY_MAPPING:
                        print(f"Warning: Unknown class {class_name} in {json_file}")
                        continue
                    
                    category_id = CATEGORY_MAPPING[class_name]

                    required_keys = ["x1", "y1", "x2", "y2"]
                    if not all(key in label for key in required_keys):
                        print(f"Warning: Missing bbox coordinates in {json_file}")
                        continue
                        
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height
                    )

                    if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                        f.write(
                            f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )
                    else:
                        print(f"Warning: Invalid bbox values in {json_file}")
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


def load_yolo_bbox(txt_path):
    """Load YOLO format bounding boxes"""
    bboxes = []
    class_labels = []

    if not os.path.exists(txt_path):
        return [], []
        
    with open(txt_path, "r") as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 5:
                class_label = int(data[0])
                x_center, y_center, width, height = map(float, data[1:])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_label)
    return bboxes, class_labels


def save_yolo_bbox(bboxes, class_labels, txt_path):
    """Save YOLO format bounding boxes"""
    with open(txt_path, "w") as f:
        for bbox, class_label in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(
                f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
