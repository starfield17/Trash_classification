import cv2
import torch
import serial
import threading
import time
import subprocess
import sys
import os
import json
import base64
from openai import OpenAI
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# Global control variables
DEBUG_WINDOW = True  # Whether to display the debug window
ENABLE_SERIAL = True  # Whether to enable serial communication
CONF_THRESHOLD = 0.6  # Confidence threshold
API_CALL_INTERVAL = 0.5  # API call interval (seconds)

ENABLE_CROP = False  # Whether to enable frame cropping
X_CROP = 720  # Cropped width
Y_CROP = 720  # Cropped height

# Serial port configuration
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_SERIAL_VALUE = 255


class SerialManager:
    def __init__(self):
        self.stm32_port = None
        self.is_running = True
        self.last_stm32_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1  # Minimum send interval (seconds)

        # Current frame detection results
        self.current_detections = []

        # Serial port send queue related
        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.MAX_QUEUE_SIZE = 50  # Maximum queue length

        # Garbage counting and recording related
        self.garbage_count = 0
        self.detected_items = []

        # Anti-duplicate counting and stability detection related
        self.last_count_time = 0
        self.COUNT_COOLDOWN = 5.0
        self.is_counting_locked = False
        self.last_detected_type = None

        # Stability detection related
        self.current_detection = None
        self.detection_start_time = 0
        self.STABILITY_THRESHOLD = 1.0
        self.stable_detection = False
        self.detection_lost_time = 0
        self.DETECTION_RESET_TIME = 0.5

        # Classification mapping
        waste_classifier = WasteClassifier()
        self.zero_mapping = max(waste_classifier.class_names.keys()) + 1
        print(f"Class 0 will be mapped to: {self.zero_mapping}")

        # Initialize STM32 serial port
        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(
                    STM32_PORT, STM32_BAUD, timeout=0.1, write_timeout=0.1
                )
                print(f"STM32 serial port initialized: {STM32_PORT}")

                # Start queue processing thread
                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("Serial port send queue processing thread started")
            except Exception as e:
                print(f"STM32 serial port initialization failed: {str(e)}")
                self.stm32_port = None

    def check_detection_stability(self, garbage_type):
        """Check detection stability"""
        current_time = time.time()

        # If a new object type is detected, or if the detection interruption exceeds the reset time
        if garbage_type != self.current_detection or (
            current_time - self.detection_lost_time > self.DETECTION_RESET_TIME
            and self.detection_lost_time > 0
        ):
            # Reset detection status
            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        # If the stable recognition time has been reached
        if (
            current_time - self.detection_start_time >= self.STABILITY_THRESHOLD
            and not self.stable_detection
        ):
            self.stable_detection = True
            return True
        return self.stable_detection

    def can_count_new_garbage(self, garbage_type):
        """Check if new garbage can be counted"""
        current_time = time.time()

        # Check stability
        if not self.check_detection_stability(garbage_type):
            return False
        # If it's a new garbage type, reset the lock state
        if garbage_type != self.last_detected_type:
            self.is_counting_locked = False
            self.last_detected_type = garbage_type
        # Check if within cooldown period
        if self.is_counting_locked:
            if current_time - self.last_count_time >= self.COUNT_COOLDOWN:
                self.is_counting_locked = False  # Release lock
            else:
                return False
        return True

    def update_garbage_count(self, garbage_type):
        """Update garbage count"""
        if not self.can_count_new_garbage(garbage_type):
            return
        self.garbage_count += 1
        self.detected_items.append(
            {
                "count": self.garbage_count,
                "type": garbage_type,
                "quantity": 1,
                "status": "Correct", # Assuming "正确" means correct
            }
        )

        # Update counting-related status
        self.last_count_time = time.time()
        self.is_counting_locked = True
        self.last_detected_type = garbage_type

    def clear_detections(self):
        """Clear detection results for the current frame"""
        self.current_detections = []

    def add_detection(self, class_id, center_x, center_y):
        """Add new detection result to the temporary list"""
        self.current_detections.append((class_id, center_x, center_y))

    def queue_processor_thread(self):
        """Queue processing thread, periodically retrieves data from the queue and sends it"""
        print("Queue processing thread started")
        while self.is_running:
            try:
                self._process_queue_batch()
            except Exception as e:
                print(f"Queue processing exception: {str(e)}")
            # Control processing frequency
            time.sleep(self.MIN_SEND_INTERVAL / 2)

    def _process_queue_batch(self):
        """Process data in the send queue"""
        # 1. Check serial port status
        if not self.stm32_port:
            print("Error: Serial port object does not exist")
            return
        if not self.stm32_port.is_open:
            try:
                print("Attempting to reopen serial port...")
                self.stm32_port.open()
                print("Serial port reopened successfully")
            except Exception as e:
                print(f"Failed to reopen serial port: {str(e)}")
                return
        # 2. Control sending frequency
        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return  # Send interval not reached, wait for next processing
        # 3. Get one piece of data from the queue
        data_to_send = None
        with self.queue_lock:
            if self.send_queue:
                data_to_send = self.send_queue.pop(0)
        if not data_to_send:
            return
        try:
            # 4. Assemble data packet
            data = bytes(
                [
                    data_to_send["class_id"],
                    data_to_send["x"],
                    data_to_send["y"],
                ]
            )

            # 5. Send data
            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            self.last_stm32_send_time = current_time

            if DEBUG_WINDOW:
                print("\n----- Serial Send Detailed Data [DEBUG] -----")
                print(f"Hexadecimal data: {' '.join([f'0x{b:02X}' for b in data])}")
                print("Original data packet structure:")
                print(
                    f"  [0] 0x{data[0]:02X} - Class ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})"
                )
                print(
                    f"  [1] 0x{data[1]:02X} - X coordinate ({data_to_send['orig_x']} -> {data_to_send['x']})"
                )
                print(
                    f"  [2] 0x{data[2]:02X} - Y coordinate ({data_to_send['orig_y']} -> {data_to_send['y']})"
                )
                print(f"Total packet length: {len(data)} bytes, actual written: {bytes_written} bytes")
                print(
                    f"Original class ID: {data_to_send['orig_class']} (decimal) -> {data_to_send['class_id']} (sent value)"
                )
                print(
                    f"Original coordinates: ({data_to_send['orig_x']}, {data_to_send['orig_y']}) -> Scaled: (0x{data_to_send['x']:02X}, 0x{data_to_send['y']:02X})"
                )
                print(
                    f"Data waiting time in queue: {current_time - data_to_send['timestamp']:.3f} seconds"
                )
                print("-" * 50)
        except serial.SerialTimeoutException:
            print("Serial write timeout, device may not be responding")
            # Timed-out data is put back into the queue to prevent data loss
            with self.queue_lock:
                self.send_queue.insert(0, data_to_send)
        except Exception as e:
            print(f"Serial send error: {str(e)}")
            # Also consider putting back into the queue on send failure
            with self.queue_lock:
                # Only put back into the queue if retry count does not exceed limit
                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # Retry at most 3 times
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send)
                    print(f"Data will be resent, attempt {retry_count}")

    def send_to_stm32(self, class_id, center_x, center_y):
        """Send data to STM32, using a queue to ensure reliable transmission"""
        # Check serial port and queue thread status
        if not self.stm32_port or not self.stm32_port.is_open:
            print("Warning: Serial port not initialized or not open, cannot send data")
            return False
        # Check if queue thread is running
        if hasattr(self, "queue_thread") and not self.queue_thread.is_alive():
            print("Warning: Queue processing thread not running, attempting to restart")
            try:
                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("Serial port send queue processing thread restarted")
            except Exception as e:
                print(f"Failed to start queue thread: {str(e)}")
                return False
        # Ensure data is within valid range
        if class_id == 0:
            mapped_class_id = self.zero_mapping
        else:
            mapped_class_id = class_id
        # Force all values to be within the 0-255 range
        mapped_class_id = min(255, max(0, mapped_class_id))
        x_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_x * MAX_SERIAL_VALUE / CAMERA_WIDTH))
        )
        y_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_y * MAX_SERIAL_VALUE / CAMERA_HEIGHT))
        )

        # Add to send queue
        with self.queue_lock:
            # Limit queue size to avoid excessive memory usage
            if len(self.send_queue) >= 10:  # Reduce queue size limit to avoid too much backlog
                # Keep the newest data, discard old data
                self.send_queue = self.send_queue[-9:]
                print("Warning: Send queue is full, discarding old data")
            # Add data to the queue, including original and scaled values
            self.send_queue.append(
                {
                    "class_id": mapped_class_id,
                    "x": x_scaled,
                    "y": y_scaled,
                    "timestamp": time.time(),
                    "orig_x": center_x,
                    "orig_y": center_y,
                    "orig_class": class_id,
                    "retry": 0,  # Initial retry count
                }
            )
        return True

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("Cleaning up resources...")

        # Wait for queue processing thread to finish
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("Queue processing thread terminated")
        # Close serial port
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            print("Serial port closed")


class QwenDetector:
    def __init__(self, api_key=None, enable_area_sorting=True, send_interval=1.0):
        """Initialize Qwen2.5-VL detector
        
        Args:
            api_key: Dashscope API key
            enable_area_sorting: Whether to enable processing sorted by area
            send_interval: Send interval time (seconds)
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen2.5-vl-72b-instruct"
        self.waste_classifier = WasteClassifier()
        self.serial_manager = SerialManager()
        
        # Detection control parameters
        self.last_detection_time = 0
        self.detection_interval = API_CALL_INTERVAL  # Control API call frequency
        self.last_detection_dict = {}
        self.min_position_change = 20
        
        # Category color mapping
        self.colors = {
            0: (86, 180, 233),   # Kitchen waste - Blue
            1: (230, 159, 0),    # Recyclable waste - Orange
            2: (240, 39, 32),    # Hazardous waste - Red
            3: (0, 158, 115),    # Other waste - Green
        }
        
        # Save current detection results for display between frames
        self.current_detections = []
        self.is_detection_in_progress = False
        
        # New: Variables related to area sorting and sequential processing
        self.enable_area_sorting = enable_area_sorting
        self.send_interval = send_interval  # Send interval time
        self.processing_queue = []  # Stores pending detection results
        self.is_processing = True   # Whether the queue is being processed
        self.queue_lock = threading.Lock()  # Queue lock
        
        # If area sorting is enabled, start the processing thread
        if self.enable_area_sorting:
            self.process_thread = threading.Thread(target=self._process_queue)
            self.process_thread.daemon = True
            self.process_thread.start()
            print(f"Detection sequential processing thread started, send interval: {self.send_interval} seconds")
        
        print(f"Qwen2.5-VL detector initialized, using model: {self.model}")
        print(f"Area priority sorting: {'Enabled' if self.enable_area_sorting else 'Disabled'}")

    def _calculate_area(self, x1, y1, x2, y2):
        """Calculate the area of the detection box"""
        return abs((x2 - x1) * (y2 - y1))
    
    def _process_queue(self):
        """Process detection results in the queue and send them sequentially"""
        while self.is_processing:
            detection_to_process = None
            
            with self.queue_lock:
                if self.processing_queue:
                    detection_to_process = self.processing_queue.pop(0)
            
            if detection_to_process:
                # Unpack detection result
                class_id = detection_to_process['class_id']
                center_x = detection_to_process['center_x']
                center_y = detection_to_process['center_y']
                confidence = detection_to_process['confidence']
                display_text = detection_to_process['display_text']
                area = detection_to_process['area']
                
                print(f"\nSending detection: {display_text}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Center point position: ({center_x}, {center_y})")
                print(f"Target area: {area} pixels^2")
                print("-" * 30)
                
                # Send to serial port
                self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                self.serial_manager.update_garbage_count(display_text)
                
                # Wait for the set interval time
                time.sleep(self.send_interval)
            else:
                # Sleep briefly when the queue is empty to avoid high CPU usage
                time.sleep(0.1)
    
    def _should_send_detection(self, class_id, center_x, center_y, confidence):
        """Determine if the current detection result should be sent"""
        current_time = time.time()
        
        # Check confidence
        if confidence < CONF_THRESHOLD:
            return False
            
        # If it's a new class, send
        if class_id not in self.last_detection_dict:
            self.last_detection_dict[class_id] = {
                "position": (center_x, center_y),
                "time": current_time,
            }
            return True
            
        # Check position change
        last_pos = self.last_detection_dict[class_id]["position"]
        if (
            abs(center_x - last_pos[0]) > self.min_position_change
            or abs(center_y - last_pos[1]) > self.min_position_change
        ):
            # Significant position change
            self.last_detection_dict[class_id] = {
                "position": (center_x, center_y),
                "time": current_time,
            }
            return True
            
        # Check time interval
        last_detection_time = self.last_detection_dict[class_id]["time"]
        if current_time - last_detection_time > 3.0:  # If not updated for more than 3 seconds, also send
            self.last_detection_dict[class_id] = {
                "position": (center_x, center_y),
                "time": current_time,
            }
            return True
            
        return False

    def detect(self, frame):
        """Detect garbage using Qwen2.5-VL API"""
        current_time = time.time()
        original_frame = frame.copy()
        
        # Always draw current detection results (even without new API calls)
        if DEBUG_WINDOW:
            frame = self.draw_detections(frame)
        
        # Control API call frequency
        if current_time - self.last_detection_time < self.detection_interval or self.is_detection_in_progress:
            return frame
        
        # Mark detection start
        self.is_detection_in_progress = True
        self.last_detection_time = current_time
        
        # Start a thread for API calls to avoid blocking the main thread
        detection_thread = threading.Thread(
            target=self._perform_detection, 
            args=(original_frame,)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        return frame
    
    def _perform_detection(self, frame):
        """Perform API detection in a separate thread"""
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare API request
            prompt = (
                "Please detect all visible garbage in the image. Identify each garbage object as one of the following four categories (Note: all vegetables/fruits count as kitchen waste):"
                "0: Kitchen waste, 1: Recyclable waste, 2: Hazardous waste, 3: Other waste."
                "Please be sure to detect all garbage objects in the image and do not miss any."
                "For each detected object, provide the class ID, confidence, and bounding box coordinates (top-left and bottom-right corners, and center point)."
                "Return the results in JSON format as follows, where the detections array can contain detection results for multiple objects:"
                "```json\n"
                "{\n"
                "  \"detections\": [\n"
                "    {\n"
                "      \"class_id\": 0,\n"
                "      \"confidence\": 0.95,\n"
                "      \"x1\": 100,\n"
                "      \"y1\": 200,\n"
                "      \"x2\": 300,\n"
                "      \"y2\": 400,\n"
                "      \"center_x\": 200,\n"
                "      \"center_y\": 300\n"
                "    },\n"
                "    {\n"
                "      \"class_id\": 1,\n"
                "      \"confidence\": 0.87,\n"
                "      \"x1\": 450,\n"
                "      \"y1\": 150,\n"
                "      \"x2\": 550,\n"
                "      \"y2\": 250,\n"
                "      \"center_x\": 500,\n"
                "      \"center_y\": 200\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
                "Please detect as many garbage objects in the image as possible, and only return the JSON data. Do not add any other descriptions or explanations."
            )
            
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
            
            # Extract API response text
            response_text = response.choices[0].message.content
            
            # Try to parse JSON from the response
            try:
                # Find JSON pattern
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find content within curly braces
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
                
                # Clear old detection results
                self.current_detections = []
                
                # Collect all detected objects
                all_detections = []
                
                # Process detected objects
                if 'detections' in result and len(result['detections']) > 0:
                    for det in result['detections']:
                        class_id = int(det.get('class_id', 0))
                        confidence = float(det.get('confidence', 0.0))
                        
                        # Get coordinates
                        if 'center_x' in det and 'center_y' in det:
                            center_x = int(det['center_x'])
                            center_y = int(det['center_y'])
                        else:
                            # Calculate center point
                            x1 = int(det.get('x1', 0))
                            y1 = int(det.get('y1', 0))
                            x2 = int(det.get('x2', 0))
                            y2 = int(det.get('y2', 0))
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                        
                        # Get bounding box coordinates
                        if all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                            x1 = int(det['x1'])
                            y1 = int(det['y1'])
                            x2 = int(det['x2'])
                            y2 = int(det['y2'])
                        else:
                            # If no bounding box is provided, create a box around the center point
                            box_size = 100
                            x1 = max(0, center_x - box_size // 2)
                            y1 = max(0, center_y - box_size // 2)
                            x2 = min(frame.shape[1], center_x + box_size // 2)
                            y2 = min(frame.shape[0], center_y + box_size // 2)
                        
                        # Get garbage classification information
                        category_id, description = self.waste_classifier.get_category_info(class_id)
                        display_text = f"{category_id}({description})"
                        
                        # Calculate area
                        area = self._calculate_area(x1, y1, x2, y2)
                        
                        # Create detection object
                        detection_obj = {
                            'class_id': class_id,
                            'confidence': confidence,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'center_x': center_x,
                            'center_y': center_y,
                            'display_text': display_text,
                            'area': area  # Add area information
                        }
                        
                        # Save detection object for subsequent processing
                        all_detections.append(detection_obj)
                        
                        # Save detection result for drawing
                        self.current_detections.append(detection_obj)
                    
                    # Process detection results
                    if self.enable_area_sorting and all_detections:
                        # Sort by area from largest to smallest
                        all_detections.sort(key=lambda x: x['area'], reverse=True)
                        
                        # Clear current processing queue
                        with self.queue_lock:
                            self.processing_queue = []
                            
                            # Add detection object to the processing queue
                            for det_obj in all_detections: # Renamed det to det_obj to avoid conflict
                                if self._should_send_detection(
                                    det_obj['class_id'],
                                    det_obj['center_x'],
                                    det_obj['center_y'],
                                    det_obj['confidence']
                                ):
                                    self.processing_queue.append(det_obj)
                                    
                                    # Debug information
                                    if DEBUG_WINDOW:
                                        print(f"Added to queue: {det_obj['display_text']}, Area: {det_obj['area']}")
                    else:
                        # If area sorting is not enabled, process in the original way
                        for det_obj in all_detections: # Renamed det to det_obj to avoid conflict
                            if self._should_send_detection(
                                det_obj['class_id'],
                                det_obj['center_x'],
                                det_obj['center_y'],
                                det_obj['confidence']
                            ):
                                print(f"\nSending detection: {det_obj['display_text']}")
                                print(f"Confidence: {det_obj['confidence']:.2%}")
                                print(f"Center point position: ({det_obj['center_x']}, {det_obj['center_y']})")
                                print("-" * 30)
                                
                                # Send to serial port
                                self.serial_manager.send_to_stm32(det_obj['class_id'], det_obj['center_x'], det_obj['center_y'])
                                self.serial_manager.update_garbage_count(det_obj['display_text'])
                
                else:
                    print("No garbage objects detected")
                    
            except json.JSONDecodeError as je:
                print(f"Error parsing JSON from API response: {je}")
                print(f"Response text: {response_text}")
                
        except Exception as e:
            print(f"API request error: {str(e)}")
        
        finally:
            # Mark detection complete
            self.is_detection_in_progress = False
    
    def draw_detections(self, frame):
        """Draw current detection results on the frame"""
        if not self.current_detections:
            return frame
            
        for det in self.current_detections:
            class_id = det['class_id']
            color = self.colors.get(class_id, (255, 255, 255)) # Default to white if color not found
            
            # Draw bounding box
            cv2.rectangle(frame, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
            
            # Draw center point
            cv2.circle(frame, (det['center_x'], det['center_y']), 5, (0, 255, 0), -1) # Green center point
            
            # Draw label, including area information
            if 'area' in det:
                label = f"{det['display_text']} {det['confidence']:.2f} A:{det['area']}"
            else:
                label = f"{det['display_text']} {det['confidence']:.2f}"
                
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (det['x1'], det['y1'] - th - 10), (det['x1'] + tw + 10, det['y1']), color, -1)
            cv2.putText(frame, label, (det['x1'] + 5, det['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # White text
        
        return frame
    
    def set_send_interval(self, interval):
        """Set send interval time (seconds)"""
        if interval > 0:
            self.send_interval = interval
            print(f"Send interval time has been set to {interval} seconds")
    
    def set_area_sorting(self, enable):
        """Enable or disable area sorting feature"""
        # If the status changes
        if enable != self.enable_area_sorting:
            self.enable_area_sorting = enable
            
            # If area sorting is enabled, but the processing thread does not exist or is not running
            if enable and (not hasattr(self, 'process_thread') or not self.process_thread.is_alive()):
                self.is_processing = True
                self.process_thread = threading.Thread(target=self._process_queue)
                self.process_thread.daemon = True
                self.process_thread.start()
                print("Detection sequential processing thread started")
            
            print(f"Area priority sorting has been {'Enabled' if enable else 'Disabled'}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up QwenDetector resources...")
        
        # Stop processing thread
        if hasattr(self, 'is_processing'):
            self.is_processing = False
            
        # Wait for processing thread to finish
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0)
                print("Detection processing thread terminated")
            except Exception as e:
                print(f"Error terminating processing thread: {str(e)}")
        
        # Clean up serial manager
        if hasattr(self, "serial_manager"):
            try:
                self.serial_manager.cleanup()
            except Exception as e:
                print(f"Error cleaning up serial manager: {str(e)}")

def main():
    # Initialize detector
    api_key = os.getenv("DASHSCOPE_API_KEY")  # Can be changed to a structure like api_key='your_api_key_here', to hardcode the api-key into the code
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY environment variable not set. Please ensure the API key is set correctly.")
        print("You can set the environment variable using: export DASHSCOPE_API_KEY=your_key")
        return
        
    #detector = QwenDetector(api_key=api_key, enable_area_sorting=True, send_interval=2.0)
    detector = QwenDetector(api_key=api_key, enable_area_sorting=False)
    # Find camera
    cap = find_camera()
    if not cap:
        return
    
    print("\nSystem starting:")
    print("- Camera ready")
    print(f"- Debug window: {'On' if DEBUG_WINDOW else 'Off'}")
    print(f"- Serial output: {'On' if ENABLE_SERIAL else 'Off'}")
    print(f"- API call interval: {API_CALL_INTERVAL} seconds")
    print(f"- Frame cropping: {'On ({X_CROP}x{Y_CROP})' if ENABLE_CROP else 'Off'}")
    print("- Press 'q' to exit the program")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break
            
            # If cropping is enabled, crop the frame
            if ENABLE_CROP:
                frame = crop_frame(frame, target_width=X_CROP, target_height=Y_CROP, mode='center')
                
            # Process the current frame
            processed_frame = detector.detect(frame)
            
            if DEBUG_WINDOW:
                window_name = "Qwen_VL_detect"
                cv2.imshow(window_name, processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProgram exited normally")
                    break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, program exiting")
    finally:
        # Clean up resources
        if hasattr(detector, "cleanup"): # Check if detector has cleanup before calling
            detector.cleanup()
        if cap: # Check if cap is not None
            cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
