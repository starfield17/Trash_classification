# yolo_autolabel.py
import os
import cv2
import json
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO # Use ultralytics library for YOLO models
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yolo_autolabeling.log') # Log file name changed
    ]
)
logger = logging.getLogger(__name__)

class YOLOAutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize the YOLO-based auto labeler

        Args:
            model_path: Path to the trained YOLO model file (.pt)
            confidence_threshold: Minimum confidence to keep a detection
        """
        try:
            self.model = YOLO(model_path) # Load the YOLO model
            # Assume the model's classes are ordered as follows. Adjust if your model differs.
            # You can usually find the class order in the model's associated .yaml file or training configuration.
            self.model_class_names = self.model.names
            logger.info(f"Loaded YOLO model from: {model_path}")
            logger.info(f"Model class names: {self.model_class_names}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise # Reraise the exception to stop execution if model loading fails

        self.confidence_threshold = confidence_threshold

        # Define the desired output category names and mapping from model class ID
        # IMPORTANT: Adjust the mapping based on your specific trained model's class IDs
        # Example: If your model has {0: 'kitchen', 1: 'recyclable', 2: 'hazardous', 3: 'other'}
        # Ensure the keys match the integer class IDs predicted by your YOLO model.
        self.category_names = {
            0: "Kitchen waste",    # Key 0 corresponds to model's class ID 0
            1: "Recyclable waste", # Key 1 corresponds to model's class ID 1
            2: "Hazardous waste",  # Key 2 corresponds to model's class ID 2
            3: "Other waste",      # Key 3 corresponds to model's class ID 3
            # Add more mappings if your model has more classes you want to map
        }

        # Create reverse mapping for convenience if needed, and colors
        self.category_mapping = {v: k for k, v in self.category_names.items()}
        self.category_colors = {
            0: (86, 180, 233),    # Kitchen waste - Blue
            1: (230, 159, 0),     # Recyclable waste - Orange
            2: (240, 39, 32),     # Hazardous waste - Red
            3: (0, 158, 115),     # Other waste - Green
        }

        logger.info(f"YOLOAutoLabeler initialized.")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Output category mapping: {self.category_names}")

    def process_image(self, img_path, output_dir, viz_dir=None):
        """Process a single image using YOLO and create JSON label file

        Args:
            img_path: Path to the input image
            output_dir: Directory to save the JSON label file
            viz_dir: Directory to save visualization (if None, no visualization)

        Returns:
            tuple: (result_message, status, detection_count)
        """
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                return f"Failed to read image: {img_path}", "failed", 0

            # Get image dimensions
            img_height, img_width = img.shape[:2]

            # Create output path for JSON file
            output_path = output_dir / f"{img_path.stem}.json"

            # Check if JSON already exists
            if output_path.exists():
                return f"Skipped (already exists): {output_path}", "skipped", 0

            # Call YOLO model to get detections
            detections = self.detect_objects(img)

            # Save the JSON file
            if not detections:
                self.save_empty_json(output_path)
                result_msg = f"No valid detections: {img_path.name}"
                detection_count = 0
            else:
                self.save_json(output_path, detections, img_width, img_height)
                result_msg = f"Created {output_path.name} with {len(detections)} detections"
                detection_count = len(detections)

            # Generate visualization if requested
            if viz_dir is not None:
                viz_path = viz_dir / f"{img_path.stem}_labeled.jpg"
                self.visualize_detections(img, detections, viz_path)

            return result_msg, "success", detection_count

        except Exception as e:
            logger.exception(f"Error processing {img_path.name}")
            return f"Error processing {img_path.name}: {str(e)}", "failed", 0

    def detect_objects(self, img):
        """Use the loaded YOLO model to detect objects

        Args:
            img: Image (NumPy array)

        Returns:
            list: List of valid detections formatted for saving/visualization
        """
        valid_detections = []
        try:
            # Perform inference
            results = self.model(img, conf=self.confidence_threshold, verbose=False) # verbose=False to reduce console output

            # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # get coordinates
                    conf = float(box.conf[0]) # get confidence
                    cls = int(box.cls[0]) # get class ID

                    # Map the detected class ID to the desired category name
                    category_name = self.category_names.get(cls)

                    if category_name is not None:
                        # Store the detection using the desired category name
                        valid_detections.append({
                            "name": category_name, # Use the mapped name
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": conf
                        })
                    else:
                        logger.debug(f"Skipping detection with unknown class ID: {cls} (Confidence: {conf:.2f})")

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            # Optionally re-raise or handle differently

        return valid_detections


    def save_empty_json(self, output_path):
        """Save an empty JSON file with no detections"""
        data = {"labels": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_json(self, output_path, detections, img_width, img_height):
        """Save detections to JSON file in the required format

        Args:
            output_path: Path to save the JSON file
            detections: List of detection objects from detect_objects
            img_width: Image width in pixels
            img_height: Image height in pixels
        """
        labels = []
        for det in detections:
            # Basic validation and boundary checking
            try:
                x1 = max(0, int(det.get('x1', 0)))
                y1 = max(0, int(det.get('y1', 0)))
                x2 = min(img_width, int(det.get('x2', img_width)))
                y2 = min(img_height, int(det.get('y2', img_height)))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"Skipping invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                    continue

                label = {
                    # Save name as is (should be the mapped category name)
                    # Convert to lower if required by downstream tasks
                    "name": det.get('name', 'unknown').lower(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(det.get('confidence', 0.0))
                }
                labels.append(label)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing detection for JSON: {e}")
                continue

        # Create the final JSON structure
        data = {"labels": labels}

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def visualize_detections(self, img, detections, output_path):
        """Generate visualization of the detections

        Args:
            img: Input image (NumPy array)
            detections: List of detection objects from detect_objects
            output_path: Path to save the visualization
        """
        vis_img = img.copy()

        for det in detections:
            name = det.get('name', 'unknown') # This is the category name, e.g., "Kitchen waste"
            x1 = max(0, int(det.get('x1', 0)))
            y1 = max(0, int(det.get('y1', 0)))
            x2 = min(vis_img.shape[1], int(det.get('x2', vis_img.shape[1])))
            y2 = min(vis_img.shape[0], int(det.get('y2', vis_img.shape[0])))

            if x2 <= x1 or y2 <= y1:
                continue

            # Get category ID and color using the category name
            category_id = self.category_mapping.get(name) # Look up ID from name
            if category_id is None:
                 logger.warning(f"Invalid category name '{name}' found during visualization.")
                 color = (255, 255, 255) # Default white color
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
        """Verify all JSON files in the directory for correct format

        Args:
            directory: Directory containing JSON files

        Returns:
            dict: Results of verification
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
                        # Basic check, assumes img dims aren't known here, just checks relation
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
                results['errors'].append((str(json_file), f"Error reading/parsing: {str(e)}"))

        return results


def main():
    """Main function to process command line arguments and run the labeler"""
    parser = argparse.ArgumentParser(description='Auto-label images using a local YOLO model')
    parser.add_argument('--input_dir', required=True, help='Directory containing unlabeled images')
    parser.add_argument('--model_path', required=True, help='Path to the trained YOLO model (.pt file)')
    # Removed output_dir argument
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process in parallel')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold for YOLO detection')
    # Removed visualize and viz_dir arguments
    # Removed verify argument
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
    if model_path.suffix != ".pt":
         logger.warning(f"Model file {args.model_path} does not have a .pt extension. Ensure it's a valid YOLO model.")

    # Create labeler instance
    try:
        labeler = YOLOAutoLabeler(
            model_path=str(model_path), # Pass model path
            confidence_threshold=args.confidence
        )
    except Exception as e:
        logger.error(f"Failed to initialize YOLOAutoLabeler: {e}")
        return # Exit if initialization fails

    # Process input directory
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    # --- Hardcoded output and visualization directories ---
    base_dir = Path('.') # Current working directory
    output_dir = base_dir / "output" # Fixed output directory name
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Visualization is always enabled ---
    viz_dir = base_dir / "viz" # Fixed visualization directory name
    viz_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Visualizations will be saved to {viz_dir}")
    # --- End of changes for directories and visualization ---

    # Get list of image files
    image_extensions = args.extensions.lower().split(',')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}"))) # Also check uppercase extensions

    # Remove duplicates just in case
    image_files = list(set(image_files))

    if not image_files:
        logger.warning(f"No image files found in {input_dir} with extensions {args.extensions}")
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
                viz_dir # Pass viz_dir directly (now always enabled)
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
                    else: # status == "failed"
                        failed += 1
                        pbar.set_postfix({
                            "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                        })
                        logger.error(f"Failed job: {result}") # Log the error message for failed jobs

                    if status != "failed":
                         logger.info(result) # Log success/skip messages
                    pbar.update(1)

                except Exception as e:
                    # Catch exceptions from future.result() itself
                    logger.error(f"Error retrieving result from thread: {e}")
                    failed += 1
                    pbar.set_postfix({
                        "Success": successful, "Empty": empty, "Skipped": skipped, "Failed": failed
                    })
                    pbar.update(1)

    # --- Verification is always enabled ---
    logger.info("Verifying generated JSON files...")
    verification_results = labeler.verify_json_files(output_dir)
    logger.info(f"Verification results: {verification_results['valid']} valid, {verification_results['invalid']} invalid")
    if verification_results['errors']:
        logger.warning(f"{verification_results['invalid']} invalid files found.")
        # Log first few errors for inspection
        errors_to_show = min(5, len(verification_results['errors']))
        logger.info(f"First {errors_to_show} verification errors:")
        for i, (file, error) in enumerate(verification_results['errors'][:errors_to_show]):
            logger.info(f"  File: {file} | Error: {error}")
    else:
        logger.info("All generated JSON files passed verification.")
    # --- End of changes for verification ---

    logger.info(f"Processing complete:")
    logger.info(f"  - Total Images: {len(image_files)}")
    logger.info(f"  - Successful: {successful} ({empty} with no detections)")
    logger.info(f"  - Skipped (already labeled): {skipped}")
    logger.info(f"  - Failed: {failed}")

if __name__ == "__main__":
    main()
