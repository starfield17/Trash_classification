# fast_R_CNN_autolabel.py
import os
import cv2
import json
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

# Import PyTorch and Faster R-CNN required libraries
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision import transforms as T
from torchvision.transforms import functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fastcnn_autolabeling.log')
    ]
)
logger = logging.getLogger(__name__)

class FastCNNAutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.5, model_type="resnet50_fpn", iou_threshold=0.5):
        """Initialize Faster R-CNN based auto-labeler

        Args:
            model_path: Faster R-CNN model file path (.pth)
            confidence_threshold: Minimum confidence to keep detection results
            model_type: Backbone network type ("resnet50_fpn", "resnet50_fpn_v2", "resnet18_fpn", or "mobilenet_v3")
            iou_threshold: IoU threshold for NMS, boxes above this value are considered overlapping
        """
        # Save IoU threshold
        self.iou_threshold = iou_threshold
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.confidence_threshold = confidence_threshold
            self.model_type = model_type
            
            # Define class names (from fastcnn_deploy.py)
            self.model_class_names = ["Kitchen waste", "Recyclable waste", "Hazardous waste", "Other waste"]
            
            # Load model
            self._load_model(model_path)
            
            logger.info(f"Loaded Faster R-CNN model: {model_path}")
            logger.info(f"Model type: {model_type}")
            logger.info(f"Model class names: {self.model_class_names}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Unable to load Faster R-CNN model {model_path}: {e}")
            raise  # Re-raise exception to stop execution if model loading fails
        
        # Define output category names and mapping from model class IDs
        self.category_names = {
            0: "Kitchen_waste",     # Kitchen waste
            1: "Recyclable_waste",  # Recyclable waste
            2: "Hazardous_waste",   # Hazardous waste
            3: "Other_waste",       # Other waste
        }
        
        # Define preprocessing transforms
        self.transforms = T.Compose([
            T.ToTensor(),
        ])
        
        # Create reverse mapping and colors
        self.category_mapping = {v: k for k, v in self.category_names.items()}
        self.category_colors = {
            0: (86, 180, 233),    # Kitchen waste - Blue
            1: (230, 159, 0),     # Recyclable waste - Orange
            2: (240, 39, 32),     # Hazardous waste - Red
            3: (0, 158, 115),     # Other waste - Green
        }
        
        logger.info(f"FastCNNAutoLabeler initialization complete.")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Output category mapping: {self.category_names}")
    
    def _get_faster_rcnn_model(self, num_classes, model_type):
        """Get different types of Faster R-CNN models
        
        Args:
            num_classes: Number of classes in the model
            model_type: Backbone network type ("resnet50_fpn", "resnet50_fpn_v2", "resnet18_fpn", or "mobilenet_v3")
            
        Returns:
            FasterRCNN: Faster R-CNN model
        """
        # num_classes needs +1 because 0 is background class
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
                pretrained=True, 
                trainable_layers=3
            )
            
            # Setup RPN anchor generator - modified to match multiple feature map layers
            anchor_generator = AnchorGenerator(
                # Specify separate anchor sizes for each feature map layer
                sizes=((32,), (64,), (128,), (256,), (512,)),
                # Repeat same aspect ratio configuration for each feature map layer
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )
            
            # Setup RoI pooling size
            roi_pooler = torch.ops.torchvision.MultiScaleRoIAlign(
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
    
    def _load_model(self, model_path):
        """Load Faster R-CNN model
        
        Args:
            model_path: Model weights file path
        """
        try:
            # Get number of classes
            num_classes = len(self.model_class_names)
            
            # Build network based on model type
            self.model = self._get_faster_rcnn_model(num_classes, self.model_type)
            
            # Load pretrained weights
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle possible key name differences
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"Failed to load model directly, trying checkpoint format: {e}")
                # Try loading checkpoint format (if it's a checkpoint with model_state_dict)
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    logger.info("Successfully loaded weights from checkpoint")
                else:
                    raise RuntimeError("Unable to load model weights, please check model file format")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_image(self, img_path, output_dir, viz_dir=None):
        """Process single image using Faster R-CNN and create JSON label file

        Args:
            img_path: Input image path
            output_dir: Directory to save JSON label files
            viz_dir: Directory to save visualization results (if None, no visualization saved)

        Returns:
            tuple: (result message, status, detection count)
        """
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                return f"Unable to read image: {img_path}", "failed", 0

            # Get image dimensions
            img_height, img_width = img.shape[:2]

            # Create output path for JSON file
            output_path = output_dir / f"{img_path.stem}.json"

            # Check if JSON file already exists
            if output_path.exists():
                return f"Skipped (already exists): {output_path}", "skipped", 0

            # Call Faster R-CNN model to get detection results
            detections = self.detect_objects(img)

            # Save JSON file
            if not detections:
                self.save_empty_json(output_path)
                result_msg = f"No valid detections: {img_path.name}"
                detection_count = 0
            else:
                self.save_json(output_path, detections, img_width, img_height)
                result_msg = f"Created {output_path.name} with {len(detections)} detections"
                detection_count = len(detections)

            # Generate visualization if needed
            if viz_dir is not None:
                viz_path = viz_dir / f"{img_path.stem}_labeled.jpg"
                self.visualize_detections(img, detections, viz_path)

            return result_msg, "success", detection_count

        except Exception as e:
            logger.exception(f"Error processing {img_path.name}")
            return f"Error processing {img_path.name}: {str(e)}", "failed", 0
    
    def _calculate_iou(self, boxA, boxB):
        """Calculate IoU (Intersection over Union) of two bounding boxes

        Args:
            boxA: First bounding box [x1, y1, x2, y2]
            boxB: Second bounding box [x1, y1, x2, y2]

        Returns:
            float: IoU value, range [0, 1]
        """
        # Determine intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Calculate intersection area
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Calculate areas of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calculate IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression (NMS), keeping highest confidence among overlapping boxes

        Args:
            detections: List of detection results
            iou_threshold: IoU threshold, boxes above this value are considered overlapping

        Returns:
            list: Detection results list after applying NMS
        """
        if len(detections) == 0:
            return []

        # Sort by confidence in descending order
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        selected_dets = []

        while sorted_dets:
            # Select detection with highest confidence
            best_det = sorted_dets.pop(0)
            selected_dets.append(best_det)

            # Filter out boxes overlapping with current best detection
            remaining_dets = []
            for det in sorted_dets:
                box1 = [best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"]]
                box2 = [det["x1"], det["y1"], det["x2"], det["y2"]]
                iou = self._calculate_iou(box1, box2)

                # Keep detection if IoU is below threshold
                if iou < iou_threshold:
                    remaining_dets.append(det)

            sorted_dets = remaining_dets

        return selected_dets

    def detect_objects(self, img):
        """Detect objects using loaded Faster R-CNN model

        Args:
            img: Image (NumPy array)

        Returns:
            list: List of valid detection results formatted for saving/visualization
        """
        valid_detections = []
        try:
            # Convert BGR to RGB (Faster R-CNN model requires RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            image_tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Extract detection results from predictions
            if len(predictions) > 0:
                prediction = predictions[0]
                
                # Get predicted bounding boxes, scores and labels
                boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                
                # Process detection results
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    # Check confidence threshold
                    if score < self.confidence_threshold:
                        continue
                    
                    # Remember label starts from 1 (0 is background), so actual class is label-1
                    class_id = int(label) - 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Map detected class ID to desired category name
                    category_name = self.category_names.get(class_id)
                    
                    if category_name is not None:
                        # Store detection result
                        valid_detections.append({
                            "name": category_name,  # Use mapped name
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": float(score)
                        })
                    else:
                        logger.debug(f"Skipping detection with unknown class ID: {class_id} (confidence: {score:.2f})")
                
                # Apply NMS, keeping only highest confidence among overlapping boxes
                valid_detections = self._apply_nms(valid_detections, iou_threshold=self.iou_threshold)
                logger.debug(f"Number of detections after NMS: {len(valid_detections)}")
            
        except Exception as e:
            logger.error(f"Error during Faster R-CNN detection: {e}")
            import traceback
            traceback.print_exc()
        
        return valid_detections
    
    def save_empty_json(self, output_path):
        """Save empty JSON file with no detection results"""
        data = {"labels": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_json(self, output_path, detections, img_width, img_height):
        """Save detection results to JSON file in required format

        Args:
            output_path: Path to save JSON file
            detections: List of detection objects from detect_objects
            img_width: Image width (pixels)
            img_height: Image height (pixels)
        """
        labels = []
        for det in detections:
            # Basic validation and boundary checks
            try:
                x1 = max(0, int(det.get('x1', 0)))
                y1 = max(0, int(det.get('y1', 0)))
                x2 = min(img_width, int(det.get('x2', img_width)))
                y2 = min(img_height, int(det.get('y2', img_height)))

                # Skip invalid bounding boxes
                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"Skipping invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                    continue

                label = {
                    # Save name as is (should be mapped category name)
                    # Convert to lowercase if needed by downstream tasks
                    "name": det.get('name', 'unknown').lower(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(det.get('confidence', 0.0))
                }
                labels.append(label)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing detection result to JSON: {e}")
                continue

        # Create final JSON structure
        data = {"labels": labels}

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def visualize_detections(self, img, detections, output_path):
        """Generate visualization of detection results

        Args:
            img: Input image (NumPy array)
            detections: List of detection objects from detect_objects
            output_path: Path to save visualization
        """
        vis_img = img.copy()

        for det in detections:
            name = det.get('name', 'unknown')  # This is category name, e.g., "Kitchen_waste"
            x1 = max(0, int(det.get('x1', 0)))
            y1 = max(0, int(det.get('y1', 0)))
            x2 = min(vis_img.shape[1], int(det.get('x2', vis_img.shape[1])))
            y2 = min(vis_img.shape[0], int(det.get('y2', vis_img.shape[0])))

            if x2 <= x1 or y2 <= y1:
                continue

            # Use category name to get category ID and color
            category_id = self.category_mapping.get(name)  # Look up ID from name
            if category_id is None:
                logger.warning(f"Invalid category name '{name}' found during visualization")
                color = (255, 255, 255)  # Default white
            else:
                color = self.category_colors.get(category_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            confidence = det.get('confidence', 0.0)
            label_text = f"{name} {confidence:.2f}"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y_start = max(y1 - label_h - 10, 0)
            label_y_end = max(y1, label_h + 10)
            text_y = max(y1 - 5, label_h + 5)

            cv2.rectangle(vis_img, (x1, label_y_start), (x1 + label_w + 10, label_y_end), color, -1)

            # Draw label text
            cv2.putText(vis_img, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save visualization image
        if not cv2.imwrite(str(output_path), vis_img):
            logger.error(f"Failed to save visualization image: {output_path}")
    
    def verify_json_files(self, directory):
        """Verify format correctness of all JSON files in directory

        Args:
            directory: Directory containing JSON files

        Returns:
            dict: Verification results
        """
        json_files = list(Path(directory).glob('*.json'))
        results = {'valid': 0, 'invalid': 0, 'errors': []}
        # Get valid names (lowercase) for checking
        valid_names_lower = {name.lower() for name in self.category_mapping.keys()}

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'labels' not in data or not isinstance(data['labels'], list):
                    results['invalid'] += 1
                    results['errors'].append((str(json_file), "Missing 'labels' array"))
                    continue

                invalid_labels = []
                for i, label in enumerate(data['labels']):
                    # Check required keys (name, x1, y1, x2, y2, confidence)
                    required_keys = ['name', 'x1', 'y1', 'x2', 'y2', 'confidence']
                    if not all(key in label for key in required_keys):
                        missing = [k for k in required_keys if k not in label]
                        invalid_labels.append((i, f"Missing required fields: {missing}"))
                        continue

                    # Check if name (lowercase) is a valid category name
                    if label.get('name') not in valid_names_lower:
                        invalid_labels.append((i, f"Invalid category name: {label.get('name')}"))
                        continue

                    # Optional: Add coordinate validation
                    try:
                        x1, y1, x2, y2 = int(label['x1']), int(label['y1']), int(label['x2']), int(label['y2'])
                        # Basic check, assuming we don't know image dimensions, only check coordinate relationships
                        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                            invalid_labels.append((i, f"Invalid coordinates: ({x1},{y1},{x2},{y2})"))
                    except (ValueError, TypeError):
                        invalid_labels.append((i, f"Non-integer coordinates"))
                    # Optional: Check confidence value
                    try:
                        conf = float(label['confidence'])
                        if not (0.0 <= conf <= 1.0):
                            invalid_labels.append((i, f"Invalid confidence value: {conf}"))
                    except (ValueError, TypeError):
                        invalid_labels.append((i, f"Non-float confidence value"))

                if invalid_labels:
                    results['invalid'] += 1
                    errors_str = ", ".join([f"Label {idx}: {err}" for idx, err in invalid_labels])
                    results['errors'].append((str(json_file), f"{len(invalid_labels)} invalid labels ({errors_str})"))
                else:
                    results['valid'] += 1

            except json.JSONDecodeError:
                results['invalid'] += 1
                results['errors'].append((str(json_file), "Invalid JSON format"))
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append((str(json_file), f"Read/parse error: {str(e)}"))

        return results


def main():
    """Main function to handle command line arguments and run the labeler"""
    parser = argparse.ArgumentParser(description='Auto-label images using local Faster R-CNN model')
    parser.add_argument('--input_dir', required=True, help='Directory containing unlabeled images')
    parser.add_argument('--model_path', required=True, help='Path to trained Faster R-CNN model (.pth file)')
    parser.add_argument('--model_type', default='resnet50_fpn', choices=['resnet50_fpn', 'resnet50_fpn_v2', 'resnet18_fpn', 'mobilenet_v3'], 
                        help='Backbone network type to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process in parallel')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for NMS to filter overlapping boxes')
    parser.add_argument('--extensions', default='.jpg,.jpeg,.png,.bmp,.webp', help='Comma-separated list of image extensions')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error(f"Model file not found: {args.model_path}")
        return
    if model_path.suffix != ".pth":
        logger.warning(f"Model file {args.model_path} does not have .pth extension. Please ensure it's a valid Faster R-CNN model.")

    # Create labeler instance
    try:
        labeler = FastCNNAutoLabeler(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            model_type=args.model_type,
            iou_threshold=args.iou_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize FastCNNAutoLabeler: {e}")
        return  # Exit if initialization fails

    # Process input directory
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    # --- Hardcoded output and visualization directories ---
    base_dir = Path('.')  # Current working directory
    output_dir = base_dir / "output"  # Fixed output directory name
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Visualization always enabled ---
    viz_dir = base_dir / "viz"  # Fixed visualization directory name
    viz_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Visualization results will be saved to {viz_dir}")
    # --- End of directory and visualization changes ---

    # Get list of image files
    image_extensions = args.extensions.lower().split(',')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))  # Also check uppercase extensions

    # Remove duplicates just in case
    image_files = list(set(image_files))

    if not image_files:
        logger.warning(f"No image files with extensions {args.extensions} found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    # Process images in batches
    successful = 0
    failed = 0
    skipped = 0
    empty = 0

    # Create futures for all images
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = []
        for img_path in image_files:
            futures.append(executor.submit(
                labeler.process_image,
                img_path,
                output_dir,
                viz_dir  # Pass viz_dir directly (now always enabled)
            ))

        # Process with progress bar
        with tqdm(total=len(image_files), desc="Labeling images") as pbar:
            for future in futures:
                try:
                    result, status, detection_count = future.result()

                    if status == "success":
                        successful += 1
                        if detection_count == 0:
                            empty += 1
                        pbar.set_postfix({
                            "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                        })
                    elif status == "skipped":
                        skipped += 1
                        pbar.set_postfix({
                            "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                        })
                    else:  # status == "failed"
                        failed += 1
                        pbar.set_postfix({
                            "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                        })
                        logger.error(f"Failed task: {result}")  # Log error message for failed task

                    if status != "failed":
                        logger.info(result)  # Log success/skip messages
                    pbar.update(1)

                except Exception as e:
                    # Catch exceptions from future.result() itself
                    logger.error(f"Error retrieving result from thread: {e}")
                    failed += 1
                    pbar.set_postfix({
                        "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                    })
                    pbar.update(1)

    # --- Verification always enabled ---
    logger.info("Verifying generated JSON files...")
    verification_results = labeler.verify_json_files(output_dir)
    logger.info(f"Verification results: {verification_results['valid']} valid, {verification_results['invalid']} invalid")
    if verification_results['errors']:
        logger.warning(f"Found {verification_results['invalid']} invalid files.")
        # Log first few errors for inspection
        errors_to_show = min(5, len(verification_results['errors']))
        logger.info(f"First {errors_to_show} verification errors:")
        for i, (file, error) in enumerate(verification_results['errors'][:errors_to_show]):
            logger.info(f"  File: {file} | Error: {error}")
    else:
        logger.info("All generated JSON files passed verification.")
    # --- End of verification changes ---

    logger.info(f"Processing complete:")
    logger.info(f"  - Total images: {len(image_files)}")
    logger.info(f"  - Successful: {successful} ({empty} with no detections)")
    logger.info(f"  - Skipped (already labeled): {skipped}")
    logger.info(f"  - Failed: {failed}")


if __name__ == "__main__":
    main()
