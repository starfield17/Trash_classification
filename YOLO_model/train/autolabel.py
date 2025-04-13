import os
import cv2
import json
import base64
import argparse
import time
import re
import logging
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_labeling.log')
    ]
)
logger = logging.getLogger(__name__)

class QwenAutoLabeler:
    def __init__(self, api_key, confidence_threshold=0.5, max_retries=3):
        """Initialize the Qwen 2.5 VL-based auto labeler
        
        Args:
            api_key: DashScope API key
            confidence_threshold: Minimum confidence to keep a detection
            max_retries: Maximum number of API call retries
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen2.5-vl-72b-instruct"
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        
        # Define the mapping from object types to categories
        self.category_mapping = {
            # 厨余垃圾 (0)
            "potato": 0,
            "daikon": 0,
            "carrot": 0,
            # 可回收垃圾 (1)
            "bottle": 1, 
            "can": 1,
            # 有害垃圾 (2)
            "battery": 2,
            "drug": 2,
            "inner_packing": 2, 
            # 其他垃圾 (3)
            "tile": 3,
            "stone": 3,
            "brick": 3,
        }
        
        # For reverse lookup - from category ID to name
        self.category_names = {
            0: "Kitchen waste",
            1: "Recyclable waste",
            2: "Hazardous waste",
            3: "Other waste",
        }
        
        # Define colors for visualization
        self.category_colors = {
            0: (86, 180, 233),    # 厨余垃圾 - Blue
            1: (230, 159, 0),     # 可回收垃圾 - Orange
            2: (240, 39, 32),     # 有害垃圾 - Red
            3: (0, 158, 115),     # 其他垃圾 - Green
        }
        
        logger.info(f"QwenAutoLabeler initialized with model: {self.model}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def process_image(self, img_path, output_dir, viz_dir=None):
        """Process a single image and create JSON label file
        
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
            
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', img)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Create output path for JSON file
            output_path = output_dir / f"{img_path.stem}.json"
            
            # Check if JSON already exists
            if output_path.exists():
                return f"Skipped (already exists): {output_path}", "skipped", 0
            
            # Call API to get detections
            detections = self.detect_objects(base64_image)
            
            # Save the JSON file
            if not detections:
                # Create empty JSON with no detections
                self.save_empty_json(output_path)
                result_msg = f"No valid detections: {img_path.name}"
                detection_count = 0
            else:
                # Create and save JSON file
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
    
    def detect_objects(self, base64_image):
        """Call Qwen 2.5 VL API to detect objects in the image with retries
        
        Args:
            base64_image: Base64 encoded image
            
        Returns:
            list: List of valid detections
        """
        prompt = (
            "Please identify objects belonging to the four main waste categories in the image and provide their locations. The categories and specific items (with broadened definitions) are:\n"
            "- **Kitchen waste:** potato, daikon, carrot (recognize them regardless of form, e.g., whole, cut, sliced).\n"
            "- **Recyclable waste:** bottle (including plastic, glass, metal bottles, *paper cups*, and similar containers), can (like soda cans).\n"
            "- **Hazardous waste:** battery, drug (*including the medicine itself and its outer packaging like paper boxes*), inner_packing (e.g., blister packs).\n"
            "- **Other waste:** tile, stone, brick.\n\n"
            "**Important:** The primary goal is to correctly classify each detected object into one of these four main waste types. While you should try to identify the specific item, accurately assigning it to the correct *main category* (Kitchen, Recyclable, Hazardous, Other) is most crucial. Minor distinctions *within* a category (e.g., potato vs. daikon) are secondary.\n\n"
            "For each detected object, provide:\n"
            "1. name: Use one of the original 11 English names (potato, daikon, carrot, bottle, can, battery, drug, inner_packing, tile, stone, brick). For example, label a paper cup as 'bottle'. Label drug packaging as 'drug'.\n"
            "2. Bounding box coordinates (x1, y1, x2, y2).\n"
            "3. Detection confidence score.\n\n"
            "Return the results strictly in JSON format:\n"
            "```json\n"
            "{\n"
            "  \"labels\": [\n"
            "    {\n"
            "      \"name\": \"potato\",\n"
            "      \"x1\": 100,\n"
            "      \"y1\": 200,\n"
            "      \"x2\": 300,\n"
            "      \"y2\": 400,\n"
            "      \"confidence\": 0.95\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"bottle\", \/\/ Could be a plastic bottle or a paper cup\n"
            "      \"x1\": 450,\n"
            "      \"y1\": 150,\n"
            "      \"x2\": 550,\n"
            "      \"y2\": 250,\n"
            "      \"confidence\": 0.87\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"drug\", \/\/ Could be the medicine or its box\n"
            "      \"x1\": 600,\n"
            "      \"y1\": 300,\n"
            "      \"x2\": 700,\n"
            "      \"y2\": 350,\n"
            "      \"confidence\": 0.90\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "If no relevant objects are found, return an empty `labels` array. Provide only the JSON output."
        )

        retries = 0
        while retries <= self.max_retries:
            try:
                # Send API request
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                
                # Extract JSON from response
                response_text = response.choices[0].message.content
                
                # Parse JSON from response text
                try:
                    # First try to find JSON block
                    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find content between curly braces
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            json_str = response_text
                    
                    # Clean up to ensure valid JSON
                    json_str = json_str.strip()
                    if json_str.startswith("```") and json_str.endswith("```"):
                        json_str = json_str[3:-3].strip()
                    
                    result = json.loads(json_str)
                    
                    # Extract and filter valid detections
                    valid_detections = []
                    
                    if 'labels' in result and isinstance(result['labels'], list):
                        for det in result['labels']:
                            # Skip if required fields are missing
                            if not all(key in det for key in ['name', 'x1', 'y1', 'x2', 'y2']):
                                continue
                            
                            # Get object name and check if it's in our mapping
                            name = det.get('name', '').lower()
                            if name not in self.category_mapping:
                                logger.debug(f"Skipping detection with name '{name}' - not in category mapping")
                                continue
                            
                            # Check confidence threshold if available
                            confidence = det.get('confidence', 1.0)
                            if confidence < self.confidence_threshold:
                                logger.debug(f"Skipping detection with low confidence: {confidence}")
                                continue
                            
                            # Add valid detection
                            valid_detections.append(det)
                    
                    return valid_detections
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parsing error: {je}")
                    logger.debug(f"Response text: {response_text}")
                    
                    # Increment retry counter
                    retries += 1
                    if retries <= self.max_retries:
                        logger.info(f"Retrying API call ({retries}/{self.max_retries})...")
                        time.sleep(1)  # Add delay before retry
                    continue
                    
            except Exception as e:
                logger.warning(f"API request error: {str(e)}")
                
                # Increment retry counter
                retries += 1
                if retries <= self.max_retries:
                    logger.info(f"Retrying API call ({retries}/{self.max_retries})...")
                    time.sleep(1)  # Add delay before retry
                continue
        
        # If we get here, all retries failed
        logger.error(f"Failed to get valid response after {self.max_retries} retries")
        return []
    
    def save_empty_json(self, output_path):
        """Save an empty JSON file with no detections"""
        data = {"labels": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_json(self, output_path, detections, img_width, img_height):
        """Save detections to JSON file in the required format
        
        Args:
            output_path: Path to save the JSON file
            detections: List of detection objects
            img_width: Image width in pixels
            img_height: Image height in pixels
        """
        # Create the labels list in the required format
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
                    "name": det.get('name', '').lower(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
                
                # Add confidence if available (optional)
                if 'confidence' in det:
                    label["confidence"] = float(det.get('confidence', 1.0))
                
                labels.append(label)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing detection: {e}")
                continue
        
        # Create the final JSON structure
        data = {"labels": labels}
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def visualize_detections(self, img, detections, output_path):
        """Generate visualization of the detections
        
        Args:
            img: Input image
            detections: List of detection objects
            output_path: Path to save the visualization
        """
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            # Get coordinates and name
            name = det.get('name', '').lower()
            x1 = max(0, int(det.get('x1', 0)))
            y1 = max(0, int(det.get('y1', 0)))
            x2 = min(vis_img.shape[1], int(det.get('x2', vis_img.shape[1])))
            y2 = min(vis_img.shape[0], int(det.get('y2', vis_img.shape[0])))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Get category ID and color
            category_id = self.category_mapping.get(name, 0)
            color = self.category_colors.get(category_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            confidence = det.get('confidence', 1.0)
            category_name = self.category_names.get(category_id, "Unknown")
            label_text = f"{name} ({category_name}) {confidence:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_img, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save visualization image
        cv2.imwrite(str(output_path), vis_img)
    
    def verify_json_files(self, directory):
        """Verify all JSON files in the directory for correct format
        
        Args:
            directory: Directory containing JSON files
            
        Returns:
            dict: Results of verification
        """
        json_files = list(Path(directory).glob('*.json'))
        results = {'valid': 0, 'invalid': 0, 'errors': []}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for required structure
                if 'labels' not in data or not isinstance(data['labels'], list):
                    results['invalid'] += 1
                    results['errors'].append((str(json_file), "Missing 'labels' array"))
                    continue
                
                # Check each label
                invalid_labels = []
                for i, label in enumerate(data['labels']):
                    # Check for required fields
                    if not all(key in label for key in ['name', 'x1', 'y1', 'x2', 'y2']):
                        invalid_labels.append((i, "Missing required fields"))
                        continue
                    
                    # Check if name is valid
                    if label['name'].lower() not in self.category_mapping:
                        invalid_labels.append((i, f"Invalid name: {label['name']}"))
                        continue
                
                if invalid_labels:
                    results['invalid'] += 1
                    results['errors'].append((str(json_file), f"{len(invalid_labels)} invalid labels"))
                else:
                    results['valid'] += 1
                    
            except json.JSONDecodeError:
                results['invalid'] += 1
                results['errors'].append((str(json_file), "Invalid JSON format"))
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append((str(json_file), f"Error: {str(e)}"))
        
        return results

def main():
    """Main function to process command line arguments and run the labeler"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Auto-label images using Qwen 2.5 VL model')
    parser.add_argument('--input_dir', required=True, help='Directory containing unlabeled images')
    parser.add_argument('--output_dir', help='Directory to save labeled JSON files (defaults to same as input)')
    parser.add_argument('--api_key', help='DashScope API key (can also use DASHSCOPE_API_KEY env variable)')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process in parallel')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    parser.add_argument('--viz_dir', help='Directory to save visualization images (defaults to "viz" subfolder)')
    parser.add_argument('--retry', type=int, default=3, help='Number of retries for API calls')
    parser.add_argument('--verify', action='store_true', help='Verify generated JSON files')
    parser.add_argument('--extensions', default='.jpg,.jpeg,.png', help='Comma-separated list of image extensions')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize API client
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set DASHSCOPE_API_KEY env variable or use --api_key")
        return
    
    # Create labeler instance
    labeler = QwenAutoLabeler(
        api_key=api_key, 
        confidence_threshold=args.confidence,
        max_retries=args.retry
    )
    
    # Process input directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup visualization directory if needed
    viz_dir = None
    if args.visualize:
        viz_dir = Path(args.viz_dir) if args.viz_dir else output_dir / "viz"
        viz_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Visualizations will be saved to {viz_dir}")
    
    # Get list of image files
    image_extensions = args.extensions.split(',')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
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
                viz_dir if args.visualize else None
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
                            pbar.set_postfix({"Success": successful, "Empty": empty})
                        else:
                            pbar.set_postfix({"Success": successful, "Detected": detection_count})
                    elif status == "skipped":
                        skipped += 1
                        pbar.set_postfix({"Skipped": skipped})
                    else:
                        failed += 1
                        pbar.set_postfix({"Failed": failed})
                        
                    logger.info(result)
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    failed += 1
                    pbar.update(1)
    
    # Verify generated JSON files if requested
    if args.verify:
        logger.info("Verifying generated JSON files...")
        verification_results = labeler.verify_json_files(output_dir)
        logger.info(f"Verification results: {verification_results['valid']} valid, {verification_results['invalid']} invalid")
        if verification_results['errors']:
            logger.info("First 5 errors:")
            for i, (file, error) in enumerate(verification_results['errors'][:5]):
                logger.info(f"  {file}: {error}")
    
    logger.info(f"Processing complete:")
    logger.info(f"  - Successful: {successful} ({empty} empty)")
    logger.info(f"  - Skipped: {skipped}")
    logger.info(f"  - Failed: {failed}")

if __name__ == "__main__":
    main()
