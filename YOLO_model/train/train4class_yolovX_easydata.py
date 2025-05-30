import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np
from pathlib import Path
import gc
import torch
from concurrent.futures import ThreadPoolExecutor

select_model = "yolo12s.pt"  # Selected model, defaults to yolo12s, can be changed
datapath = "./label"  # Modify according to the actual situation

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

# Helper function to check a single image and its label file
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
            return None # Indicate failure
        # Check if image dimensions are reasonable
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            print(f"Warning: Image dimensions too small: {img_file}")
            return None # Indicate failure
    except Exception as e:
        print(f"Warning: Error reading image {img_file}: {e}")
        return None # Indicate failure

    # Check if label file exists and is valid
    if os.path.exists(json_file):
        # Ensure label file is indeed a JSON file
        if not json_file.lower().endswith(".json"):
            print(f"Warning: Incorrect label file extension (should be .json): {json_file}")
            return None # Indicate failure

        # Validate JSON file content (first check validity)
        if not validate_json_file(json_file):
             # validate_json_file already printed a warning
            return None # Indicate failure

        # Validate JSON file structure (second check structure)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            if "labels" not in label_data:
                print(f"Warning: Invalid label file structure (missing 'labels' key): {json_file}")
                return None # Indicate failure
            # If all checks pass, return the image file name
            return img_file
        except Exception as e:
            # Catch errors during the structure check read
            print(f"Warning: Error processing label file {json_file} (structure check): {e}")
            return None # Indicate failure
    else:
        print(f"Warning: Corresponding label file not found: {json_file}")
        return None # Indicate failure


def check_and_clean_dataset(data_dir):
    """Checks dataset integrity and cleans invalid data (Parallelized Version)"""
    print("Checking dataset integrity (parallel)...")
    image_extensions = (".jpg", ".jpeg", ".png")

    try:
        # Check if data_dir exists before listing
        if not os.path.isdir(data_dir):
             print(f"Error: Data directory does not exist or is not a directory: {data_dir}")
             return []
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"Error: Cannot list directory {data_dir}: {e}")
        return []

    image_files = [
        f for f in all_files if f.lower().endswith(image_extensions)
    ]
    print(f"Found {len(image_files)} potential image files")
    if not image_files:
        print("No supported image files found in the directory.")
        return []

    valid_pairs = []
    futures = []
    # Determine max_workers, leave some cores free for other tasks
    # Use at least 1 worker even if cpu_count is None or 1
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using up to {max_workers} worker threads for checking...")

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit check task for each image file
        for img_file in image_files:
            futures.append(executor.submit(_check_single_file, img_file, data_dir))

        # Collect results as they complete
        for future in futures:
            try:
                result = future.result() # Get the return value from _check_single_file
                if result:  # If the helper function returned an img_file name (success)
                    valid_pairs.append(result)
            except Exception as exc:
                # Catch potential exceptions during future execution/result retrieval
                # Although _check_single_file handles most internal errors
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
        "names": {0: "Kitchen Waste", 1: "Recyclable Waste", 2: "Hazardous Waste", 3: "Other Waste"},
        "nc": 4,  # Changed to 4 major categories
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

def try_create_symlink(src, dst):
    """Try to create a symbolic link and return True if successful, False otherwise."""
    try:
        # Create the symbolic link
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception as e:
        print(f"Warning: Could not create symbolic link from {src} to {dst}: {e}")
        return False


def process_split(split_name, files, data_dir, use_symlinks=True):
    """
    Processes image copying/linking and label conversion for a given dataset split.
    
    Args:
        split_name: Name of the split (train, val, test)
        files: List of image filenames to process
        data_dir: Source directory containing the original files
        use_symlinks: Whether to try using symbolic links instead of copying (default: True)
    """
    print(f"\nProcessing {split_name} split...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    # Ensure target directories exist
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)
    
    # First file - try to create a symlink to test if it works
    symlinks_working = False
    if use_symlinks and files:
        test_img = os.path.join(data_dir, files[0])
        test_dst = os.path.join(split_img_dir, "test_symlink_" + files[0])
        symlinks_working = try_create_symlink(test_img, test_dst)
        
        # Clean up the test link
        if os.path.exists(test_dst):
            os.remove(test_dst)
    
    print(f"{split_name}: Using {'symbolic links' if symlinks_working else 'file copying'} for dataset preparation")

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file)
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt")

        # Process the image and label files
        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                # Handle the image file - try symlink first if enabled
                if symlinks_working:
                    if not try_create_symlink(src_img, dst_img):
                        # If symlink creation fails, fall back to copying
                        shutil.copy2(src_img, dst_img)
                else:
                    # Use direct copy if symlinks aren't working
                    shutil.copy2(src_img, dst_img)
                
                # Convert the label (no need to change this part)
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
    """Prepare dataset - supports using symbolic links instead of copying files to save space
    
    Args:
        data_dir: Source directory containing image and label files
        valid_pairs: List of valid image filenames
        use_symlinks: Whether to try using symbolic links instead of copying files (default: True)
    """
    # Ensure validation set has at least 10 images
    if len(valid_pairs) < 15:
        raise ValueError(
            f"Insufficient number of valid data pairs ({len(valid_pairs)}). At least 15 images are required."
        )
    
    # Clean existing directories - improved cleaning method
    print("\nCleaning existing train/val/test directories...")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), split)
        if os.path.exists(split_path):
            print(f"Deleting {split_path} directory...")
            try:
                # Try to delete the directory directly
                shutil.rmtree(split_path)
            except OSError as e:
                print(f"Cannot delete directory {split_path} directly, error: {e}")
                print(f"Attempting to delete files and subdirectories individually...")
                
                # Delete files and directories individually
                for root, dirs, files in os.walk(split_path, topdown=False):
                    # Delete files first
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Warning: Cannot delete file {file_path}: {e}")
                    
                    # Then delete empty directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                            print(f"Deleted directory: {dir_path}")
                        except Exception as e:
                            print(f"Warning: Cannot delete directory {dir_path}: {e}")
                
                # Finally, try to delete the main directory
                try:
                    os.rmdir(split_path)
                    print(f"Successfully deleted {split_path}")
                except Exception as e:
                    print(f"Warning: Cannot delete main directory {split_path}: {e}")
                    print(f"Will attempt to continue processing...")
        
        # Recreate directory structure
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

    # Use ThreadPoolExecutor to process each dataset split in parallel
    print("Starting parallel processing of dataset splits...")
    # Determine number of workers, reserve some cores for other tasks
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each split, including use_symlinks parameter
        futures = {
            executor.submit(process_split, split_name, files, data_dir, use_symlinks): split_name
            for split_name, files in splits.items()
        }

        # Wait for all tasks to complete and check for exceptions
        for future in futures:
            split_name = futures[future]
            try:
                future.result()  # Wait for task to complete and raise exception (if any)
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

    # Calculate center point coordinates and width/height

    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Ensure values are within the 0-1 range

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
            print(f"Warning: No corresponding image file found for: {json_file}")
            return False
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image: {img_path}")
            return False
        img_height, img_width = img.shape[:2]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Mapping from fine-grained categories to major categories

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
                print(f"Warning: No 'labels' key in {json_file}")
                return False
            for label in data["labels"]:
                try:
                    if "name" not in label:
                        print(f"Warning: No 'name' field in label data in {json_file}")
                        continue
                    class_name = label["name"]
                    if class_name not in category_mapping:
                        print(f"Warning: Unknown class {class_name} in {json_file}")
                        continue
                    # Directly use major category ID

                    category_id = category_mapping[class_name]

                    required_keys = ["x1", "y1", "x2", "y2"]
                    if not all(key in label for key in required_keys):
                        print(f"Warning: Missing bbox coordinates in {json_file}")
                        continue
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height
                    )

                    if all(
                        0 <= val <= 1 for val in [x_center, y_center, width, height]
                    ):
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

def save_quantized_models(weights_dir, data_yaml_path):
    """Load the best model and save FP16 version"""
    import shutil
    
    best_pt_path = os.path.join(weights_dir, 'best.pt')
    if not os.path.exists(best_pt_path):
        print(f"Error: {best_pt_path} not found, cannot save quantized model.")
        return

    print(f"\nLoading best model from {best_pt_path}...")
    try:
        model = YOLO(best_pt_path)
    except Exception as e:
        print(f"Error loading model {best_pt_path}: {e}")
        return

    # FP32 weight saving part removed

    # Create FP16 model
    print("\nCreating FP16 model...")
    # Copy original model
    fp16_model_path = os.path.join(weights_dir, 'best_fp16.pt')
    shutil.copy(best_pt_path, fp16_model_path)
    
    # Load copy and convert to FP16
    fp16_model = YOLO(fp16_model_path)
    if hasattr(fp16_model, 'model'):
        fp16_model.model = fp16_model.model.half()
        # Save converted model
        fp16_model.save(fp16_model_path)
        print(f"FP16 model saved to {fp16_model_path}")
    else:
        print("Cannot convert to FP16 model: model structure not as expected")
    
    # Clean up memory
    del fp16_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Export TorchScript model (optional but useful)
    try:
        print("\nAttempting to export TorchScript format model...")
        torchscript_results = model.export(format='torchscript')
        if hasattr(torchscript_results, 'saved_model'):
            ts_path = torchscript_results.saved_model
            print(f"TorchScript model exported to: {ts_path}")
            
            # Move to target location (if export location is not in weights_dir)
            if os.path.dirname(ts_path) != weights_dir:
                ts_target_path = os.path.join(weights_dir, 'best.torchscript')
                shutil.copy(ts_path, ts_target_path)
                print(f"TorchScript model copied to: {ts_target_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    print("\nModel export and save operations complete!")
    
    # Clean up memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_yolo(use_augmentation=False, use_mixed_precision=False, config="default", resume=False):
    """
    YOLO training configuration, adds data augmentation, mixed-precision training, and multiple training configuration options.
    Supports CPU and GPU training.
    Args:
        use_augmentation (bool): Whether to enable data augmentation, defaults to False.
        use_mixed_precision (bool): Whether to enable mixed-precision training, defaults to False.
        config (str): Training configuration mode, default is 'default'. Options include:
            - 'default': Default configuration
            - 'large_dataset': Optimized configuration for large datasets
            - 'small_dataset': Optimized configuration for small datasets
            - 'focus_accuracy': Optimized configuration for detection accuracy
            - 'focus_speed': Optimized configuration for training speed
            - 'servermode': Optimized configuration for rented servers (completely overrides original config) 
    """
    model = YOLO(select_model)  # Load pre-trained YOLO model weights
    num_workers = max(1, min(os.cpu_count() - 2, 8))
    device = "cpu"
    if torch.cuda.is_available():
        device = "0"
    if device == "cpu":
        batch_size = 4  # Reduce batch size
        workers = max(1, min(os.cpu_count() - 1, 4))  # Reduce number of workers
        use_mixed_precision = False  # CPU does not support mixed-precision training
    else:
        batch_size = 10  # Original batch size
        workers = num_workers
    # Basic training parameters

    train_args = {
        "data": "data.yaml",  # Dataset configuration file path
        "epochs": 120,  # Total number of training epochs
        "imgsz": 640,  # Input image size
        "batch": batch_size,  # Batch size adjusted according to device
        "workers": workers,  # Number of worker threads adjusted according to device
        "device": device,  # Automatically selected training device
        "patience": 15,  # Number of epochs for early stopping tolerance
        "save_period": 5,  # Save model every N epochs
        "exist_ok": True,  # Whether to overwrite if results directory exists
        "project": os.path.dirname(os.path.abspath(__file__)),  # Project directory for training results
        "name": "runs/train",  # Name of the training run
        "optimizer": "AdamW",  # Optimizer type
        "lr0": 0.0005,  # Initial learning rate
        "lrf": 0.01,  # Ratio of final learning rate to initial learning rate
        "momentum": 0.937,  # Optimizer momentum parameter
        "weight_decay": 0.0005,  # Weight decay (regularization) coefficient
        "warmup_epochs": 10,  # Number of epochs for warmup phase
        "warmup_momentum": 0.5,  # Momentum during warmup phase
        "warmup_bias_lr": 0.05,  # Learning rate for bias during warmup phase
        "box": 4.0,  # Bounding box regression loss weight
        "cls": 2.0,  # Classification loss weight
        "dfl": 1.5,  # Distributed Focal Loss weight
        "close_mosaic": 0,  # Whether to close mosaic data augmentation
        "nbs": 64,  # Nominal batch size
        "overlap_mask": False,  # Whether to use overlap mask
        "multi_scale": True,  # Whether to enable multi-scale training
        "single_cls": False,  # Whether to treat all classes as a single class
        "rect": True,
        "cache": True,
    }
    train_args.update({"resume": resume})
    # Update training parameters based on configuration mode

    if config == "large_dataset":
        train_args.update(
            {
                "batch": 32 if device == "0" else 4,  # Use 32 for GPU, 4 for CPU
                "lr0": 0.001,
                "epochs": 150,
                "patience": 30,
            }
        )
    elif config == "small_dataset":
        train_args.update(
            {
                "batch": 16 if device == "0" else 4,  # Use 16 for GPU, 4 for CPU
                "lr0": 0.0001,
                "weight_decay": 0.001,
                "warmup_epochs": 15,
            }
        )
    elif config == "focus_accuracy":
        train_args.update(
            {
                "imgsz": 640,
                "box": 7.5,
                "cls": 4.0,
                "dfl": 3.0,
                "patience": 20,
                "batch": 16 if device == "0" else 4,  # Use 16 for GPU, 4 for CPU
                "epochs": 300,
                "lr0": 0.001,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "dropout": 0.1, # Add dropout to reduce overfitting
            }
        )
    elif config == "focus_speed":
        train_args.update(
            {
                "imgsz": 512,
                "epochs": 150,
                "patience": 30,
                "batch": 48 if device == "0" else 4,  # Use 48 for GPU, 4 for CPU
            }
        )
    elif config == "servermode":
        server_worker = int(os.cpu_count() / 2)
        train_args.update({
            # Retain accuracy optimization parameters from focus_accuracy
            "box": 7.5,                # Increase bounding box loss weight
            "cls": 4.0,                # Increase classification loss weight
            "dfl": 3.0,                # Increase DFL loss weight
            "patience": 15,            # Maintain high tolerance to ensure optimal accuracy
            "epochs": 150,             # Maintain a longer training cycle
            "dropout": 0.1,            # Retain dropout to prevent overfitting
            # Server performance optimization parameters
            "imgsz": 640,             
            "batch": 32,               # Increase batch size
            "lr0": 0.001,              # Learning rate from focus_accuracy
            "lrf": 0.01,               # Learning rate decay from focus_accuracy
            "weight_decay": 0.0005,    # Retain weight_decay parameter
            "optimizer": "AdamW",      # Use AdamW optimizer
            "workers": server_worker ,  # Fully utilize CPU cores
            "device": "0",             # Ensure GPU is used
            "half": True,              # Force enable half-precision training
            "cache": "ram",            # Use RAM cache for acceleration
            "cos_lr": True,            # Use cosine learning rate schedule
            "warmup_epochs": 10,       # Maintain sufficient warmup
            #"close_mosaic": 50,        # Bug here, disable for now
            "overlap_mask": True,      # Enable overlap mask
            "save_period": 5,         # Periodically save checkpoints
            "multi_scale": True,       # Multi-scale training enhances generalization
        })
        
    elif config != "default":
        print(f"Warning: Unrecognized configuration mode '{config}', using default configuration.")
    # Data augmentation parameters
    if use_augmentation:
        augmentation_args = {
            "augment": True,  # Enable data augmentation
            "degrees": 10.0,  # Random rotation angle range
            "scale": 0.2,  # Random scaling ratio range
            "fliplr": 0.2,  # Probability of random horizontal flip
            "flipud": 0.2,  # Probability of random vertical flip
            #"hsv_h": 0.03,  # Random hue adjustment range
            #"hsv_s": 0.2,  # Random saturation adjustment range
            #"hsv_v": 0.1,  # Random brightness adjustment range
            #"mosaic": 0.1,  # Mosaic augmentation ratio
            #"mixup": 0.1,  # Mixup augmentation ratio
            #"copy_paste": 0.05,  # Copy-paste augmentation ratio
        }
        train_args.update(augmentation_args)
    else:
        # Force disable all data augmentation
        no_augment_args = {
            "augment": False,  # Disable data augmentation
            "degrees": 0.0,  # Disable rotation
            "scale": 0.0,  # Disable scaling
            "fliplr": 0.0,  # Disable horizontal flip
            "flipud": 0.0,  # Disable vertical flip
            "hsv_h": 0.0,  # Disable hue adjustment
            "hsv_s": 0.0,  # Disable saturation adjustment
            "hsv_v": 0.0,  # Disable brightness adjustment
            "mosaic": 0,  # Disable mosaic
            "mixup": 0,  # Disable mixup
            "copy_paste": 0,  # Disable copy-paste
        }
        train_args.update(no_augment_args)
    # Enable mixed-precision training (only on GPU)

    if use_mixed_precision and device == "0":
        train_args.update({"half": True})
    else:
        train_args.update({"half": False})
    try:
        print(f"\nUsing device: {'GPU' if device == '0' else 'CPU'}")
        print(f"Batch size: {train_args['batch']}")
        print(f"Mixed-precision training: {'Enabled' if train_args.get('half', False) else 'Disabled'}\n")
        results = model.train(**train_args)
        
        # Force saving a best.pt if training was resumed
        if resume:
            print("\nDetected resume=True, ensuring final model is saved to best.pt...")
            run_dir = results.save_dir if hasattr(results, 'save_dir') else train_args.get('project', '') + '/' + train_args.get('name', 'runs/train')
            weights_dir = os.path.join(run_dir, 'weights')
            last_pt_path = os.path.join(weights_dir, 'last.pt')
            best_pt_path = os.path.join(weights_dir, 'best.pt')
            
            if os.path.exists(last_pt_path):
                # Copy last.pt to best.pt, ensuring there's always an up-to-date model for post-processing
                if not os.path.exists(best_pt_path) or resume:
                    print(f"Copying {last_pt_path} to {best_pt_path}...")
                    shutil.copy2(last_pt_path, best_pt_path)
                    print(f"Successfully saved final model to {best_pt_path}")
        
        return results
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def main():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        data_dir = datapath
        
        # 1. Check dataset
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(data_dir)
        if not valid_pairs:
             print("No valid data pairs found. Exiting.")
             return # Exit if no data
        gc.collect()

        # 2. Create configuration file
        print("\nStep 2: Creating data.yaml...")
        create_data_yaml()
        # Define data_yaml_path here for later use
        project_path = os.path.dirname(os.path.abspath(__file__))
        data_yaml_path = os.path.join(project_path, "data.yaml")

        # 3. Prepare dataset - use symlinks
        print("\nStep 3: Preparing dataset with symbolic links...")
        try:
             # Enable symbolic link option
             train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs, use_symlinks=True)
             gc.collect()
             if val_size < 5: # Check validation set size after preparation
                 print(f"Warning: Validation set size ({val_size}) is less than 5. INT8 calibration might be suboptimal.")
        except ValueError as ve:
             print(f"Error during dataset preparation: {ve}")
             return # Exit if dataset prep fails critically
        
        print("\nStep 4: Starting training...")
        # Define your desired training config and resume flag
        training_config = "servermode" # Example: Use 'servermode' config
        resume_training = False       # Example: Start a new training run
        results = train_yolo(
            use_augmentation=True,      # Example: Enable augmentation
            use_mixed_precision=True,   # Example: Enable mixed precision (often handled by config)
            config=training_config,
            resume=resume_training
        )
        
        if results: # Check if training returned results (didn't fail)
            print("\nStep 5: Saving different precision models based on best.pt...")
            # Determine the path to the weights directory from the results object if possible
            # Fallback to constructing the path if needed
            weights_dir = None
            if hasattr(results, 'save_dir'):
                 weights_dir = os.path.join(results.save_dir, 'weights')
                 print(f"Found weights directory from results: {weights_dir}")
            else:
                 # Fallback: Construct the expected path (less reliable if name/project changed)
                 # Try to get run_name and project from train_args if they were set there
                 # This part assumes train_args is accessible or its relevant parts are passed/known
                 # For simplicity, if train_args is not directly accessible here,
                 # you might need to pass 'name' and 'project' from train_yolo or reconstruct them.
                 # Assuming train_args was defined in a scope accessible here or its values are known:
                 # run_name = train_args.get('name', 'runs/train') # Get name from train_args if possible
                 # project = train_args.get('project', project_path)

                 # A more robust fallback if train_args isn't available:
                 # This requires knowing the default project and name structure.
                 default_project = project_path # project_path defined earlier
                 default_run_name_base = "runs/train" # Default name from train_args

                 run_dirs = sorted(Path(default_project).glob(f"{Path(default_run_name_base).name}*/"), key=os.path.getmtime, reverse=True)
                 if run_dirs:
                     latest_run_dir = run_dirs[0]
                     weights_dir = os.path.join(latest_run_dir, 'weights')
                     print(f"Constructed weights directory (fallback): {weights_dir}")
                 else:
                     print(f"Warning: Could not determine weights directory automatically (fallback).")


            if weights_dir and os.path.isdir(weights_dir):
                 save_quantized_models(weights_dir, data_yaml_path)
            else:
                 print(f"Warning: Weights directory '{weights_dir}' not found. Skipping post-training save.")
        else:
            print("\nTraining did not complete successfully or was interrupted. Skipping post-training model saving.")
    except Exception as e:
        print(f"\nAn error occurred in the main execution flow: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
    finally: # Ensure final cleanup happens
        print("\nScript finished. Cleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()
