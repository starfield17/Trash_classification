import os
import shutil
import json
import gc
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# Data path, modify according to the actual situation
datapath = "./label"

# Select model type
MODEL_TYPE = "resnet50_fpn"  # Standard versions: "resnet50_fpn" & "resnet50_fpn_v2", Lightweight version: "resnet18_fpn", Ultra-lightweight version: "mobilenet_v3"

# Four-category garbage dataset configuration
CATEGORY_MAPPING = {
    # Kitchen waste (0)
    "Kitchen_waste": 0,
    "potato": 0,
    "daikon": 0,
    "carrot": 0,
    # Recyclable waste (1)
    "Recyclable_waste": 1,
    "bottle": 1,
    "can": 1,
    # Hazardous waste (2)
    "Hazardous_waste": 2,
    "battery": 2,
    "drug": 2,
    "inner_packing": 2,
    # Other waste (3)
    "Other_waste": 3,
    "tile": 3,
    "stone": 3,
    "brick": 3,
}

# Category names
CLASS_NAMES = ["Kitchen_waste", "Recyclable_waste", "Hazardous_waste", "Other_waste"]


def validate_json_file(json_path):
    """
    Validate if the JSON file is valid.
    Returns True if valid, False if invalid.
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
        # Ensure the label file is indeed a JSON file
        if not json_file.lower().endswith(".json"):
            print(f"Warning: Label file extension incorrect (should be .json): {json_file}")
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
                print(f"Warning: Label file structure invalid (missing 'labels' key): {json_file}")
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
    """Check dataset integrity and clean invalid data (Parallelized Version)"""
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


def prepare_dataset(data_dir, valid_pairs):
    """Prepare dataset - Modify validation set split ratio"""
    # Ensure validation set has at least 10 images
    if len(valid_pairs) < 15:
        raise ValueError(
            f"Insufficient number of valid data pairs ({len(valid_pairs)}). At least 15 images are required."
        )

    # Clean existing directories
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
                print(f"Attempting to delete files and subdirectories one by one...")

                # Delete files and directories one by one
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
                    print(f"Will try to continue processing...")

        # Recreate directory structure
        print(f"Creating {split} directory structure...")
        os.makedirs(split_path, exist_ok=True)

    # Dataset split (90% train, 5% validation, 5% test)
    print("Splitting dataset into training, validation, and test sets...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.1, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)

    print("\nDataset preparation complete.")
    print(f"Training set: {len(train_files)} images, Validation set: {len(val_files)} images, Test set: {len(test_files)} images")

    return train_files, val_files, test_files


# Custom dataset class
class GarbageDataset(Dataset):
    def __init__(self, img_files, data_dir, transforms=None):
        self.img_files = img_files
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(self.data_dir, base_name + ".json")

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Read annotations
        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # Prepare target boxes and labels
        boxes = []
        labels = []
        for label in label_data.get("labels", []):
            if "name" not in label or label["name"] not in CATEGORY_MAPPING:
                continue

            # Check if bounding box coordinates exist
            required_keys = ["x1", "y1", "x2", "y2"]
            if not all(key in label for key in required_keys):
                continue

            # Get category ID and bounding box coordinates
            category_id = CATEGORY_MAPPING[label["name"]]
            x1, y1, x2, y2 = label["x1"], label["y1"], label["x2"], label["y2"]

            # Ensure coordinates are valid
            x1, x2 = max(0, min(x1, image.shape[1])), max(0, min(x2, image.shape[1]))
            y1, y2 = max(0, min(y1, image.shape[0])), max(0, min(y2, image.shape[0]))

            # Calculate width and height
            width = x2 - x1
            height = y2 - y1

            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue

            boxes.append([x1, y1, x2, y2])
            # TorchVision's object detection models require category IDs starting from 1 (0 is background)
            labels.append(category_id + 1)

        # Ensure at least one target
        if not boxes:
            # Create a very small dummy box and label
            boxes = torch.as_tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.as_tensor([1], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Build target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # Apply transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# Image transformations and augmentations
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)

            # Update bounding box coordinates
            width = image.shape[2]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes

        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_faster_rcnn_model(num_classes, model_type="resnet50_fpn"):
    """Get different types of Faster R-CNN models"""
    # num_classes needs +1 because 0 is the background class
    num_classes_with_bg = num_classes + 1

    if model_type == "resnet50_fpn":
        # Standard version: ResNet50+FPN
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    elif model_type == "resnet18_fpn":
        # Lightweight version: ResNet18+FPN
        backbone = resnet_fpn_backbone(
            'resnet18',
            weights='IMAGENET1K_V1', # Use pretrained weights
            trainable_layers=3
        )

        # Set RPN's anchor generator - modify here to match multiple feature map layers
        anchor_generator = AnchorGenerator(
            # Specify separate anchor sizes for each feature map layer
            sizes=((32,), (64,), (128,), (256,), (512,)),
            # Repeat the same aspect ratio configuration for each feature map layer
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # Set RoI pooling size
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create Faster R-CNN model
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes_with_bg,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    elif model_type == "mobilenet_v3":
        # Ultra-lightweight version: MobileNetV3
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    elif model_type == "resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    metric_logger = {}
    metric_logger["loss"] = 0
    metric_logger["loss_classifier"] = 0
    metric_logger["loss_box_reg"] = 0
    metric_logger["loss_objectness"] = 0
    metric_logger["loss_rpn_box_reg"] = 0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_box = 0.0
    running_loss_obj = 0.0
    running_loss_rpn = 0.0

    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_box = 0.0
    total_loss_obj = 0.0
    total_loss_rpn = 0.0

    start_time = time.time()

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Calculate loss
        loss_value = losses.item()
        loss_classifier = loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
        loss_box_reg = loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
        loss_objectness = loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0

        # Accumulate total loss for calculating epoch average loss
        total_loss += loss_value
        total_loss_cls += loss_classifier
        total_loss_box += loss_box_reg
        total_loss_obj += loss_objectness
        total_loss_rpn += loss_rpn_box_reg

        running_loss += loss_value
        running_loss_cls += loss_classifier
        running_loss_box += loss_box_reg
        running_loss_obj += loss_objectness
        running_loss_rpn += loss_rpn_box_reg

        if not torch.isfinite(losses):
            print(f"Warning: Loss is not a finite value: {loss_value}")
            print(f"Training will continue, but please monitor the results closely")
            # Skip this batch
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if i % print_freq == 0 and i > 0:
            avg_loss = running_loss / print_freq
            avg_loss_cls = running_loss_cls / print_freq
            avg_loss_box = running_loss_box / print_freq
            avg_loss_obj = running_loss_obj / print_freq
            avg_loss_rpn = running_loss_rpn / print_freq

            elapsed = time.time() - start_time
            print(f"Epoch: {epoch} [{i}/{len(data_loader)}]\t"
                  f"Loss: {avg_loss:.4f}\t"
                  f"Time: {elapsed:.2f}s")

            # Reset counters
            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_box = 0.0
            running_loss_obj = 0.0
            running_loss_rpn = 0.0
            start_time = time.time()

    # Calculate average loss for the epoch
    num_batches = len(data_loader)
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
    epoch_loss_cls = total_loss_cls / num_batches if num_batches > 0 else 0
    epoch_loss_box = total_loss_box / num_batches if num_batches > 0 else 0
    epoch_loss_obj = total_loss_obj / num_batches if num_batches > 0 else 0
    epoch_loss_rpn = total_loss_rpn / num_batches if num_batches > 0 else 0

    # Update and return metrics
    metric_logger["loss"] = epoch_loss
    metric_logger["loss_classifier"] = epoch_loss_cls
    metric_logger["loss_box_reg"] = epoch_loss_box
    metric_logger["loss_objectness"] = epoch_loss_obj
    metric_logger["loss_rpn_box_reg"] = epoch_loss_rpn

    return metric_logger

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()

    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_box = 0.0
    total_loss_obj = 0.0
    total_loss_rpn = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Temporarily switch to training mode to calculate loss during evaluation
            model.train()
            loss_dict = model(images, targets)
            model.eval()  # Switch back to evaluation mode immediately

            losses = sum(loss for loss in loss_dict.values())

            # Accumulate loss
            loss_value = losses.item()
            loss_classifier = loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
            loss_box_reg = loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
            loss_objectness = loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0

            total_loss += loss_value
            total_loss_cls += loss_classifier
            total_loss_box += loss_box_reg
            total_loss_obj += loss_objectness
            total_loss_rpn += loss_rpn_box_reg

    # Calculate average loss
    metrics = {}
    metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0
    metrics["loss_classifier"] = total_loss_cls / num_batches if num_batches > 0 else 0
    metrics["loss_box_reg"] = total_loss_box / num_batches if num_batches > 0 else 0
    metrics["loss_objectness"] = total_loss_obj / num_batches if num_batches > 0 else 0
    metrics["loss_rpn_box_reg"] = total_loss_rpn / num_batches if num_batches > 0 else 0

    return metrics


def save_optimized_model(model, output_dir, device, model_type):
    """Save optimized model, including different formats and precisions"""
    os.makedirs(output_dir, exist_ok=True)

    # Save base model
    model_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Export model (TorchScript format)
    try:
        # Switch to evaluation mode
        model.eval()

        # Create script model using example input
        dummy_input = [torch.rand(3, 640, 640).to(device)]
        script_model = torch.jit.trace(model, dummy_input)
        script_model_path = os.path.join(output_dir, "model_scripted.pt")
        torch.jit.save(script_model, script_model_path)
        print(f"TorchScript model saved to: {script_model_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # If CUDA is available, save half-precision model
    if device != "cpu" and torch.cuda.is_available():
        try:
            model_fp16 = model.half()
            fp16_path = os.path.join(output_dir, "model_fp16.pth")
            torch.save(model_fp16.state_dict(), fp16_path)
            print(f"FP16 model saved to: {fp16_path}")

            # Restore to FP32
            model = model.float()
        except Exception as e:
            print(f"FP16 model saving failed: {e}")

    # Export ONNX model (suitable for Raspberry Pi deployment)
    try:
        dummy_input_onnx = torch.rand(1, 3, 640, 640).to(device) # Batch size 1 for ONNX
        input_names = ["input"]
        output_names = ["boxes", "labels", "scores"]

        # Temporarily create a forward function that supports ONNX export
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model_to_wrap):
                super(ModelWrapper, self).__init__()
                self.model_to_wrap = model_to_wrap

            def forward(self, x):
                predictions = self.model_to_wrap([x]) # Model expects a list of tensors
                if predictions and isinstance(predictions, list) and len(predictions) > 0:
                    # Ensure the output format matches what ONNX expects
                    # The model's output is a list of dictionaries, one per image.
                    # For a single image input, we take the first dictionary.
                    pred_dict = predictions[0]
                    return pred_dict["boxes"], pred_dict["labels"], pred_dict["scores"]
                else:
                    # Handle cases where predictions might be empty or not in the expected format
                    # This might require returning empty tensors or raising an error
                    # For simplicity, returning empty tensors of expected types if no predictions
                    return torch.empty(0, 4, device=x.device), \
                           torch.empty(0, dtype=torch.long, device=x.device), \
                           torch.empty(0, device=x.device)


        wrapper = ModelWrapper(model)
        wrapper.eval() # Ensure wrapper is also in eval mode

        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(
            wrapper,
            dummy_input_onnx,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11, # or a higher version if needed
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, # if you want dynamic axes
                          'boxes': {0: 'num_detections'},
                          'labels': {0: 'num_detections'},
                          'scores': {0: 'num_detections'}}
        )
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Model quantization (INT8) - more suitable for running on Raspberry Pi
    try:
        # Note: Actual quantization usually requires a calibration dataset
        # This shows a simplified quantization process
        # Ensure model is on CPU for quantization
        model_cpu = model.to('cpu')
        model_cpu.eval() # Quantization requires eval mode

        # Dynamic quantization for Linear layers (common for Faster R-CNN heads)
        # For full model quantization (including convolutions), static quantization is often preferred
        # which requires a calibration dataloader.
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {torch.nn.Linear}, dtype=torch.qint8
        )
        q_path = os.path.join(output_dir, "model_quantized.pth")
        torch.save(quantized_model.state_dict(), q_path)
        print(f"Quantized model saved to: {q_path}")
        # Move model back to original device if needed (though it's the end of this function)
        model.to(device)
    except Exception as e:
        print(f"Quantized model saving failed: {e}")

    print("\nModel export complete!")
    return model_path


def main():
    """Main function"""
    try:
        # Clean memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. Check dataset
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(datapath)
        if not valid_pairs:
            print("No valid data pairs found. Exiting.")
            return
        gc.collect()

        # 2. Prepare dataset
        print("\nStep 2: Preparing dataset...")
        try:
            train_files, val_files, test_files = prepare_dataset(datapath, valid_pairs)
            gc.collect()
            if len(val_files) < 5:
                print(f"Warning: Validation set size ({len(val_files)}) is less than 5. Calibration might not be ideal.")
        except ValueError as ve:
            print(f"Error during dataset preparation: {ve}")
            return

        # 3. Create datasets and data loaders
        print("\nStep 3: Creating datasets and data loaders...")
        train_dataset = GarbageDataset(train_files, datapath, transforms=get_transform(train=True))
        val_dataset = GarbageDataset(val_files, datapath, transforms=get_transform(train=False))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Adjust batch size and worker threads based on device type
        if device.type == "cuda":
            batch_size = 8
            num_workers = min(6, os.cpu_count() // 2 if os.cpu_count() else 1)
        else:
            batch_size = 2 # Smaller batch size for CPU
            num_workers = min(2, os.cpu_count() // 2 if os.cpu_count() else 1)
            # For CPU, always ensure at least one worker thread
            num_workers = max(1, num_workers)
        print(f"Batch size: {batch_size}, Num workers: {num_workers}")


        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)) # Custom collate function for object detection
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Typically 1 for validation in object detection
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # 4. Create model
        print(f"\nStep 4: Creating {MODEL_TYPE} model...")
        model = get_faster_rcnn_model(len(CLASS_NAMES), model_type=MODEL_TYPE)
        model.to(device)

        # 5. Set optimizer and training parameters
        print("\nStep 5: Setting optimizer and training parameters...")
        # Group parameters to apply different learning rates
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when validation loss stops decreasing
            factor=0.5,      # Reduce LR by half when plateaued
            patience=3,      # Wait 3 epochs before reducing
            min_lr=1e-6      # Do not go below this value
        )

        # Number of training epochs
        num_epochs = min(max(10, len(train_files) // (10 * batch_size) if batch_size > 0 else 10), 200) # Dynamic epochs based on dataset size
        print(f"Will train for a maximum of {num_epochs} epochs, with early stopping enabled")

        # 6. Start training
        print("\nStep 6: Starting training of Faster R-CNN model...")
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_early_stop = 10  # Stop training if validation loss doesn't improve for 10 consecutive epochs
        min_delta = 0.001  # Minimum improvement threshold
        counter_early_stop = 0  # Counter for epochs without improvement
        early_stopped = False

        # Record training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        start_time_total_train = time.time()

        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = train_one_epoch(
                model, optimizer, train_loader, device, epoch
            )

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Evaluate on validation set
            val_metrics = evaluate(model, val_loader, device)

            # Update learning rate scheduler based on validation loss
            lr_scheduler.step(val_metrics['loss'])

            # Record training history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['learning_rates'].append(current_lr)

            # Print training information
            print(f"\n----- Epoch {epoch}/{num_epochs-1} Training Results -----")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Training Loss: {train_metrics['loss']:.4f}")
            print(f"  Classifier Loss: {train_metrics['loss_classifier']:.4f}")
            print(f"  Box Regression Loss: {train_metrics['loss_box_reg']:.4f}")
            print(f"  Objectness Loss: {train_metrics['loss_objectness']:.4f}")
            print(f"  RPN Box Regression Loss: {train_metrics['loss_rpn_box_reg']:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"  Classifier Loss (Val): {val_metrics['loss_classifier']:.4f}")
            print(f"  Box Regression Loss (Val): {val_metrics['loss_box_reg']:.4f}")
            print(f"  Objectness Loss (Val): {val_metrics['loss_objectness']:.4f}")
            print(f"  RPN Box Regression Loss (Val): {val_metrics['loss_rpn_box_reg']:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'history': history
            }, checkpoint_path)

            # Update best model and check for early stopping
            if val_metrics['loss'] < best_val_loss - min_delta:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_metrics['loss']:.4f}. Saving best model...")
                best_val_loss = val_metrics['loss']
                counter_early_stop = 0
                # Save best model
                best_model_path = os.path.join(output_dir, "model_best.pth")
                torch.save(model.state_dict(), best_model_path)
            else:
                counter_early_stop += 1
                print(f"Validation loss did not improve. Patience counter: {counter_early_stop}/{patience_early_stop}")

                if counter_early_stop >= patience_early_stop:
                    print(f"\nEarly stopping! Validation loss did not improve for {patience_early_stop} epochs.")
                    early_stopped = True
                    break
            gc.collect() # Collect garbage at the end of each epoch

        # Calculate total training time
        total_time_train = time.time() - start_time_total_train
        hours, remainder = divmod(total_time_train, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining {'stopped early' if early_stopped else 'completed'}! Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # If early stopped, load the best model
        if early_stopped and 'best_model_path' in locals() and os.path.exists(best_model_path):
            print(f"Loading best model (from epoch {epoch - patience_early_stop})...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        elif not early_stopped:
             print("Training completed all epochs. Using the final model state.")
        else:
            print("Early stopping triggered, but best model path not found. Using current model state.")


        # 7. Save final model and optimized deployment versions
        print("\nStep 7: Saving and optimizing model for deployment...")
        model_path = save_optimized_model(model, output_dir, device, MODEL_TYPE)

        print(f"\nTraining complete! Models saved in {output_dir} directory.")
        print(f"Final model path: {model_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("You can use the saved ONNX or TorchScript model for deployment on Raspberry Pi.")

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript execution finished. Cleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
