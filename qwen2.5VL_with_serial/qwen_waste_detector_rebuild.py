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
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from openai import OpenAI
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# Global configuration
@dataclass(frozen=True)
class Config:
    DEBUG_WINDOW: bool = True
    ENABLE_SERIAL: bool = True
    CONF_THRESHOLD: float = 0.9
    API_CALL_INTERVAL: float = 0.5
    ENABLE_CROP: bool = False
    X_CROP: int = 720
    Y_CROP: int = 720
    STM32_PORT: str = "/dev/ttyUSB0"
    STM32_BAUD: int = 115200
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    MAX_SERIAL_VALUE: int = 255


# State machine states
class SerialState(Enum):
    IDLE = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()


class DetectionState(Enum):
    IDLE = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    ERROR = auto()


class ObjectTrackingState(Enum):
    INITIAL = auto()
    TRACKING = auto()
    STABLE = auto()
    LOST = auto()


@dataclass
class Detection:
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    display_text: str
    area: int
    timestamp: float = field(default_factory=time.time)
    retry: int = 0


@dataclass
class SerialMessage:
    class_id: int
    x: int
    y: int
    timestamp: float = field(default_factory=time.time)
    orig_x: int = 0
    orig_y: int = 0
    orig_class: int = 0
    retry: int = 0


@dataclass
class TrackedObject:
    class_id: int
    position: Tuple[int, int]
    first_seen: float
    last_updated: float
    is_stable: bool = False
    is_counted: bool = False


class SerialManager:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.state = SerialState.IDLE
        self.port = None
        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.is_running = True
        self.last_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1

        # Garbage counting
        self.waste_classifier = WasteClassifier()
        self.zero_mapping = max(self.waste_classifier.class_names.keys()) + 1
        self.garbage_count = 0
        self.detected_items = []
        
        # Tracking stability
        self.tracked_objects = {}  # class_id -> TrackedObject
        self.STABILITY_THRESHOLD = 1.0
        self.COUNT_COOLDOWN = 5.0
        self.DETECTION_RESET_TIME = 0.5
        
        # Initialize serial if enabled
        if self.config.ENABLE_SERIAL:
            self._initialize_serial()
    
    def _initialize_serial(self):
        """Initialize the serial connection"""
        self.state = SerialState.CONNECTING
        try:
            self.port = serial.Serial(
                self.config.STM32_PORT,
                self.config.STM32_BAUD,
                timeout=0.1,
                write_timeout=0.1
            )
            self.state = SerialState.CONNECTED
            print(f"STM32 serial initialized: {self.config.STM32_PORT}")
            
            # Start queue processing thread
            self.queue_thread = threading.Thread(target=self._process_queue)
            self.queue_thread.daemon = True
            self.queue_thread.start()
            print("Serial queue processing thread started")
        except Exception as e:
            self.state = SerialState.ERROR
            print(f"STM32 serial initialization failed: {str(e)}")
            self.port = None
    
    def _process_queue(self):
        """Process messages in the send queue"""
        while self.is_running:
            try:
                # Sleep to control processing rate
                time.sleep(self.MIN_SEND_INTERVAL / 2)
                
                if self.state != SerialState.CONNECTED:
                    continue
                
                # Check if we need to wait before sending next message
                current_time = time.time()
                if current_time - self.last_send_time < self.MIN_SEND_INTERVAL:
                    continue
                
                # Get a message from the queue
                message = None
                with self.queue_lock:
                    if self.send_queue:
                        message = self.send_queue.pop(0)
                
                if not message:
                    continue
                
                # Send the message
                self._send_message(message)
                
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
    
    def _send_message(self, message: SerialMessage):
        """Send a message over serial"""
        if not self.port or not self.port.is_open:
            with self.queue_lock:
                self.send_queue.insert(0, message)
            return
        
        try:
            # Create data packet
            data = bytes([
                message.class_id,
                message.x,
                message.y
            ])
            
            # Send data
            bytes_written = self.port.write(data)
            self.port.flush()
            self.last_send_time = time.time()
            
            # Debug output
            if self.config.DEBUG_WINDOW:
                print("\n----- Serial Transmission Details [DEBUG] -----")
                print(f"Hex data: {' '.join([f'0x{b:02X}' for b in data])}")
                print(f"Class ID: {message.orig_class} -> {message.class_id}")
                print(f"Position: ({message.orig_x}, {message.orig_y}) -> ({message.x}, {message.y})")
                print(f"Queue wait time: {time.time() - message.timestamp:.3f} seconds")
                print("-" * 50)
                
        except serial.SerialTimeoutException:
            print("Serial write timeout, device not responding")
            # Put message back in queue
            with self.queue_lock:
                self.send_queue.insert(0, message)
        except Exception as e:
            print(f"Serial send error: {str(e)}")
            retry_count = message.retry + 1
            if retry_count <= 3:
                with self.queue_lock:
                    message.retry = retry_count
                    self.send_queue.insert(0, message)
                    print(f"Will retry send, attempt {retry_count}")
    
    def send_to_stm32(self, class_id: int, center_x: int, center_y: int) -> bool:
        """Add a detection to the send queue"""
        if self.state != SerialState.CONNECTED:
            print("Warning: Serial not connected, can't send data")
            return False
        
        # Map class_id=0 to avoid confusion with null byte
        mapped_class_id = self.zero_mapping if class_id == 0 else class_id
        
        # Scale coordinates to serial range
        x_scaled = min(
            self.config.MAX_SERIAL_VALUE,
            max(0, int(center_x * self.config.MAX_SERIAL_VALUE / self.config.CAMERA_WIDTH))
        )
        y_scaled = min(
            self.config.MAX_SERIAL_VALUE,
            max(0, int(center_y * self.config.MAX_SERIAL_VALUE / self.config.CAMERA_HEIGHT))
        )
        
        # Add to send queue
        with self.queue_lock:
            if len(self.send_queue) >= 10:
                # Keep only newest data if queue is full
                self.send_queue = self.send_queue[-9:]
                print("Warning: Send queue full, dropping older data")
            
            # Create message
            self.send_queue.append(SerialMessage(
                class_id=mapped_class_id,
                x=x_scaled,
                y=y_scaled,
                orig_x=center_x,
                orig_y=center_y,
                orig_class=class_id
            ))
        
        return True
    
    def update_tracking(self, class_id: int, center_x: int, center_y: int) -> ObjectTrackingState:
        """Update object tracking and determine the state of the tracked object"""
        current_time = time.time()
        position = (center_x, center_y)
        
        # If this is a new object type
        if class_id not in self.tracked_objects:
            self.tracked_objects[class_id] = TrackedObject(
                class_id=class_id,
                position=position,
                first_seen=current_time,
                last_updated=current_time
            )
            return ObjectTrackingState.INITIAL
        
        # Get the existing tracking record
        tracking = self.tracked_objects[class_id]
        
        # Update the position and time
        tracking.position = position
        tracking.last_updated = current_time
        
        # Check for stability
        time_tracked = current_time - tracking.first_seen
        if time_tracked >= self.STABILITY_THRESHOLD and not tracking.is_stable:
            tracking.is_stable = True
            return ObjectTrackingState.STABLE
        
        return ObjectTrackingState.TRACKING
    
    def update_garbage_count(self, class_id: int, display_text: str) -> bool:
        """Update garbage count for stable detections"""
        if class_id not in self.tracked_objects:
            return False
            
        tracking = self.tracked_objects[class_id]
        current_time = time.time()
        
        # Check if object is stable and not already counted
        if not tracking.is_stable or tracking.is_counted:
            return False
            
        # Check cooldown period
        if tracking.is_counted and current_time - tracking.last_updated < self.COUNT_COOLDOWN:
            return False
            
        # Update count
        self.garbage_count += 1
        tracking.is_counted = True
        
        # Record the detection
        self.detected_items.append({
            "count": self.garbage_count,
            "type": display_text,
            "quantity": 1,
            "status": "正确"
        })
        
        return True
    
    def can_send_update(self, class_id: int) -> bool:
        """Determine if we should send an update for this object"""
        if class_id not in self.tracked_objects:
            return True
            
        tracking = self.tracked_objects[class_id]
        current_time = time.time()
        
        # If it's been more than 3 seconds since last update, send again
        if current_time - tracking.last_updated > 3.0:
            return True
            
        # If it's a stable object that hasn't been counted, send update
        if tracking.is_stable and not tracking.is_counted:
            return True
            
        return False
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.tracked_objects.clear()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("Cleaning up resources...")
        
        # Wait for queue thread to end
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("Queue processing thread terminated")
            
        # Close serial port
        if self.port and self.port.is_open:
            self.port.close()
            print("Serial port closed")
            self.state = SerialState.DISCONNECTED


class QwenDetector:
    def __init__(self, api_key=None, enable_area_sorting=True, send_interval=1.0, config=Config()):
        """Initialize Qwen2.5-VL detector"""
        self.config = config
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen2.5-vl-72b-instruct"
        self.waste_classifier = WasteClassifier()
        self.serial_manager = SerialManager(config)
        
        # Detection state
        self.detection_state = DetectionState.IDLE
        self.last_detection_time = 0
        self.current_detections = []
        
        # Detection processing params
        self.enable_area_sorting = enable_area_sorting
        self.send_interval = send_interval
        self.processing_queue = []
        self.is_processing = True
        self.queue_lock = threading.Lock()
        
        # Class colors
        self.colors = {
            0: (86, 180, 233),   # Food waste - Blue
            1: (230, 159, 0),    # Recyclable waste - Orange
            2: (240, 39, 32),    # Hazardous waste - Red
            3: (0, 158, 115),    # Other waste - Green
        }
        
        # Start processing thread if area sorting is enabled
        if self.enable_area_sorting:
            self.process_thread = threading.Thread(target=self._process_detection_queue)
            self.process_thread.daemon = True
            self.process_thread.start()
            print(f"Detection processing thread started, send interval: {self.send_interval} seconds")
            
        print(f"Qwen2.5-VL detector initialized, using model: {self.model}")
        print(f"Area priority sorting: {'enabled' if self.enable_area_sorting else 'disabled'}")
    
    def _process_detection_queue(self):
        """Process detection queue in order of area (largest first)"""
        while self.is_processing:
            detection_to_process = None
            
            # Get next detection from queue
            with self.queue_lock:
                if self.processing_queue:
                    detection_to_process = self.processing_queue.pop(0)
            
            if detection_to_process:
                # Process the detection
                class_id = detection_to_process.class_id
                center_x = detection_to_process.center_x
                center_y = detection_to_process.center_y
                display_text = detection_to_process.display_text
                confidence = detection_to_process.confidence
                area = detection_to_process.area
                
                print(f"\nSending detection: {display_text}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Center position: ({center_x}, {center_y})")
                print(f"Target area: {area} pixels^2")
                print("-" * 30)
                
                # Send to serial and update counts
                self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                tracking_state = self.serial_manager.update_tracking(class_id, center_x, center_y)
                
                if tracking_state == ObjectTrackingState.STABLE:
                    self.serial_manager.update_garbage_count(class_id, display_text)
                
                # Wait before processing next detection
                time.sleep(self.send_interval)
            else:
                # Sleep when queue is empty
                time.sleep(0.1)
    
    def detect(self, frame):
        """Detect waste in the frame"""
        current_time = time.time()
        original_frame = frame.copy()
        
        # Always draw current detections
        if self.config.DEBUG_WINDOW:
            frame = self.draw_detections(frame)
        
        # Control API call frequency
        if (current_time - self.last_detection_time < self.config.API_CALL_INTERVAL or 
            self.detection_state == DetectionState.IN_PROGRESS):
            return frame
        
        # Update state and timestamps
        self.detection_state = DetectionState.IN_PROGRESS
        self.last_detection_time = current_time
        
        # Start detection thread
        detection_thread = threading.Thread(
            target=self._perform_detection,
            args=(original_frame,)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        return frame
    
    def _perform_detection(self, frame):
        """Perform detection using API"""
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare prompt
            prompt = (
                "请检测图像中所有可见的垃圾。将每个垃圾对象标识为以下四类之一(注意:所有蔬菜/水果都算作厨余垃圾)："
                "0: 厨余垃圾, 1: 可回收垃圾, 2: 有害垃圾, 3: 其他垃圾。"
                "请务必检测出图像中的所有垃圾物体，不要遗漏。"
                "对于每个检测到的物体，提供类别ID、置信度和边界框坐标（左上和右下角以及中心点）。"
                "以JSON格式返回结果，格式如下，其中detections数组可以包含多个物体的检测结果："
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
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
                "请尽可能多地检测出图像中的所有垃圾物体，并且只返回JSON数据，不要添加任何其他说明或解释。"
            )
            
            # API call
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
            
            # Parse response
            self._parse_api_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"API request error: {str(e)}")
            self.detection_state = DetectionState.ERROR
        
        finally:
            # Mark detection as complete
            self.detection_state = DetectionState.COMPLETE
    
    def _parse_api_response(self, response_text):
        """Parse the API response and extract detections"""
        try:
            # Find JSON content
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            # Clean up JSON string
            json_str = json_str.strip()
            if json_str.startswith("```") and json_str.endswith("```"):
                json_str = json_str[3:-3].strip()
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Clear old detections
            self.current_detections = []
            all_detections = []
            
            # Process detections
            if 'detections' in result and result['detections']:
                for det in result['detections']:
                    detection = self._process_detection_entry(det)
                    if detection:
                        all_detections.append(detection)
                        self.current_detections.append(detection)
                
                # Process detections based on area sorting
                if self.enable_area_sorting and all_detections:
                    # Sort by area (largest first)
                    all_detections.sort(key=lambda x: x.area, reverse=True)
                    
                    # Update processing queue
                    with self.queue_lock:
                        self.processing_queue = []
                        
                        # Add detections that meet criteria
                        for det in all_detections:
                            if det.confidence >= self.config.CONF_THRESHOLD and self.serial_manager.can_send_update(det.class_id):
                                self.processing_queue.append(det)
                                
                                # Debug info
                                if self.config.DEBUG_WINDOW:
                                    print(f"Added to queue: {det.display_text}, area: {det.area}")
                else:
                    # Process sequentially if area sorting disabled
                    for det in all_detections:
                        if det.confidence >= self.config.CONF_THRESHOLD and self.serial_manager.can_send_update(det.class_id):
                            print(f"\nSending detection: {det.display_text}")
                            print(f"Confidence: {det.confidence:.2%}")
                            print(f"Center position: ({det.center_x}, {det.center_y})")
                            print("-" * 30)
                            
                            # Send to serial and update tracking
                            self.serial_manager.send_to_stm32(det.class_id, det.center_x, det.center_y)
                            tracking_state = self.serial_manager.update_tracking(det.class_id, det.center_x, det.center_y)
                            
                            if tracking_state == ObjectTrackingState.STABLE:
                                self.serial_manager.update_garbage_count(det.class_id, det.display_text)
            else:
                print("No waste objects detected")
                
        except json.JSONDecodeError as je:
            print(f"Error parsing JSON in API response: {je}")
            print(f"Response text: {response_text}")
            self.detection_state = DetectionState.ERROR
    
    def _process_detection_entry(self, det):
        """Process a single detection entry from API response"""
        try:
            class_id = int(det.get('class_id', 0))
            confidence = float(det.get('confidence', 0.0))
            
            # Get coordinates
            if 'center_x' in det and 'center_y' in det:
                center_x = int(det['center_x'])
                center_y = int(det['center_y'])
            else:
                # Calculate center
                x1 = int(det.get('x1', 0))
                y1 = int(det.get('y1', 0))
                x2 = int(det.get('x2', 0))
                y2 = int(det.get('y2', 0))
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            
            # Get bounding box
            if all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                x1 = int(det['x1'])
                y1 = int(det['y1'])
                x2 = int(det['x2'])
                y2 = int(det['y2'])
            else:
                # Create a box around center point
                box_size = 100
                x1 = max(0, center_x - box_size // 2)
                y1 = max(0, center_y - box_size // 2)
                x2 = min(1280, center_x + box_size // 2)  # Assuming default width
                y2 = min(720, center_y + box_size // 2)   # Assuming default height
            
            # Get classification info
            category_name, description = self.waste_classifier.get_category_info(class_id)
            display_text = f"{category_name}({description})"
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            return Detection(
                class_id=class_id,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=center_x,
                center_y=center_y,
                display_text=display_text,
                area=area
            )
            
        except Exception as e:
            print(f"Error processing detection entry: {str(e)}")
            return None
    
    def draw_detections(self, frame):
        """Draw current detections on frame"""
        if not self.current_detections:
            return frame
            
        for det in self.current_detections:
            class_id = det.class_id
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, (det.center_x, det.center_y), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"{det.display_text} {det.confidence:.2f} A:{det.area}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (det.x1, det.y1 - th - 10), (det.x1 + tw + 10, det.y1), color, -1)
            cv2.putText(frame, label, (det.x1 + 5, det.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def set_send_interval(self, interval):
        """Set the send interval time (seconds)"""
        if interval > 0:
            self.send_interval = interval
            print(f"Send interval set to {interval} seconds")
    
    def set_area_sorting(self, enable):
        """Enable or disable area-based sorting"""
        if enable != self.enable_area_sorting:
            self.enable_area_sorting = enable
            
            if enable and (not hasattr(self, 'process_thread') or not self.process_thread.is_alive()):
                self.is_processing = True
                self.process_thread = threading.Thread(target=self._process_detection_queue)
                self.process_thread.daemon = True
                self.process_thread.start()
                print("Detection processing thread started")
            
            print(f"Area priority sorting {'enabled' if enable else 'disabled'}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up QwenDetector resources...")
        
        # Stop processing thread
        self.is_processing = False
        
        # Wait for thread to end
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
    # Initialize configuration
    config = Config()
    
    # Get API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY environment variable not set")
        print("Set with: export DASHSCOPE_API_KEY=your_key")
        return
    
    # Initialize detector
    detector = QwenDetector(api_key=api_key, enable_area_sorting=False, config=config)
    
    # Find camera
    cap = find_camera()
    if not cap:
        return
    
    print("\nSystem started:")
    print("- Camera ready")
    print(f"- Debug window: {'enabled' if config.DEBUG_WINDOW else 'disabled'}")
    print(f"- Serial output: {'enabled' if config.ENABLE_SERIAL else 'disabled'}")
    print(f"- API call interval: {config.API_CALL_INTERVAL} seconds")
    print(f"- Frame cropping: {'enabled (%dx%d)' % (config.X_CROP, config.Y_CROP) if config.ENABLE_CROP else 'disabled'}")
    print("- Press 'q' to exit")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break
            
            # Crop frame if enabled
            if config.ENABLE_CROP:
                frame = crop_frame(frame, target_width=config.X_CROP, target_height=config.Y_CROP, mode='center')
            
            # Process current frame
            processed_frame = detector.detect(frame)
            
            if config.DEBUG_WINDOW:
                window_name = "Qwen_VL_detect"
                cv2.imshow(window_name, processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProgram exited normally")
                    break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, exiting")
    finally:
        # Clean up
        detector.cleanup()
        cap.release()
        if config.DEBUG_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
