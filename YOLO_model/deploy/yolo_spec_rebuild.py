import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import subprocess
import sys
import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# Configuration
@dataclass
class Config:
    """Centralized configuration for the waste detection system."""
    # Core settings
    DEBUG_WINDOW: bool = True
    ENABLE_SERIAL: bool = True
    CONF_THRESHOLD: float = 0.9
    MODEL_PATH: str = "yolov8n_e200.pt"
    
    # Camera settings
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    
    # Serial settings
    SERIAL_PORT: str = "/dev/ttyUSB0"
    SERIAL_BAUD: int = 115200
    MAX_SERIAL_VALUE: int = 255
    
    # Detection settings
    DETECTION_INTERVAL: float = 0.5
    MIN_POSITION_CHANGE: int = 20
    SEND_INTERVAL: float = 1.0
    
    # Stability settings
    STABILITY_THRESHOLD: float = 1.0
    DETECTION_RESET_TIME: float = 0.5
    COUNT_COOLDOWN: float = 5.0

# State definitions
class DetectionState(Enum):
    """State machine states for the detection process."""
    IDLE = auto()
    DETECTING = auto()
    PROCESSING = auto()
    ERROR = auto()

class SerialState(Enum):
    """State machine states for the serial communication."""
    DISCONNECTED = auto()
    CONNECTED = auto()
    SENDING = auto()
    ERROR = auto()


class SerialManager:
    """Manages serial communication with external hardware."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = SerialState.DISCONNECTED
        self.port = None
        self.is_running = True
        self.last_send_time = 0
        self.send_queue = []
        self.queue_lock = threading.Lock()
        
        # Garbage counting
        self.garbage_count = 0
        self.detected_items = []
        self.last_count_time = 0
        self.is_counting_locked = False
        self.last_detected_type = None
        
        # Stability tracking
        self.current_detection = None
        self.detection_start_time = 0
        self.stable_detection = False
        self.detection_lost_time = 0
        
        # Class mapping
        self.waste_classifier = WasteClassifier()
        self.zero_mapping = max(self.waste_classifier.class_names.keys()) + 1
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the serial connection and processing thread."""
        if not self.config.ENABLE_SERIAL:
            return
            
        try:
            self.port = serial.Serial(
                self.config.SERIAL_PORT,
                self.config.SERIAL_BAUD,
                timeout=0.1,
                write_timeout=0.1
            )
            self.state = SerialState.CONNECTED
            print(f"Serial port initialized: {self.config.SERIAL_PORT}")
            
            # Start queue processor thread
            self.queue_thread = threading.Thread(target=self._process_queue)
            self.queue_thread.daemon = True
            self.queue_thread.start()
            print("Serial queue processor thread started")
        except Exception as e:
            print(f"Failed to initialize serial port: {e}")
            self.state = SerialState.ERROR
    
    def _process_queue(self) -> None:
        """Process items in the send queue."""
        while self.is_running:
            try:
                # Check if we can send now
                current_time = time.time()
                if current_time - self.last_send_time < 0.1:  # Minimum interval
                    time.sleep(0.05)
                    continue
                
                # Check port status
                if self.state != SerialState.CONNECTED:
                    self._attempt_reconnect()
                    time.sleep(0.1)
                    continue
                
                # Get data from queue
                data_to_send = None
                with self.queue_lock:
                    if self.send_queue:
                        data_to_send = self.send_queue.pop(0)
                
                if not data_to_send:
                    time.sleep(0.05)
                    continue
                
                # Send data
                self._send_data_packet(data_to_send, current_time)
                
            except Exception as e:
                print(f"Queue processing error: {e}")
                time.sleep(0.1)
    
    def _send_data_packet(self, data_to_send: Dict, current_time: float) -> None:
        """Send a data packet over serial."""
        try:
            self.state = SerialState.SENDING
            
            # Prepare data packet
            data = bytes([
                data_to_send["class_id"],
                data_to_send["x"],
                data_to_send["y"],
            ])
            
            # Send data
            self.port.write(data)
            self.port.flush()
            self.last_send_time = current_time
            self.state = SerialState.CONNECTED
            
            # Debug output
            if self.config.DEBUG_WINDOW:
                self._print_debug_info(data, data_to_send)
                
        except serial.SerialTimeoutException:
            print("Serial write timeout")
            self._handle_send_error(data_to_send)
        except Exception as e:
            print(f"Serial send error: {e}")
            self._handle_send_error(data_to_send)
    
    def _handle_send_error(self, data_to_send: Dict) -> None:
        """Handle errors during data sending."""
        with self.queue_lock:
            retry_count = data_to_send.get("retry", 0) + 1
            if retry_count <= 3:  # Max retry limit
                data_to_send["retry"] = retry_count
                self.send_queue.insert(0, data_to_send)
                print(f"Will retry sending, attempt {retry_count}")
            else:
                print("Maximum retry attempts reached, dropping packet")
        
        self.state = SerialState.ERROR
    
    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the serial port."""
        if not self.port:
            return
            
        try:
            print("Attempting to reconnect serial port...")
            if not self.port.is_open:
                self.port.open()
            self.state = SerialState.CONNECTED
            print("Serial port reconnected")
        except Exception as e:
            print(f"Reconnection failed: {e}")
            self.state = SerialState.ERROR
    
    def _print_debug_info(self, data: bytes, data_to_send: Dict) -> None:
        """Print debug information about the sent data."""
        print("\n----- Serial Send Data [DEBUG] -----")
        print(f"Hex data: {' '.join([f'0x{b:02X}' for b in data])}")
        print(f"Class ID: {data_to_send['orig_class']} -> {data_to_send['class_id']}")
        print(f"X coordinate: {data_to_send['orig_x']} -> {data_to_send['x']}")
        print(f"Y coordinate: {data_to_send['orig_y']} -> {data_to_send['y']}")
        print(f"Queue wait time: {time.time() - data_to_send['timestamp']:.3f}s")
        print("-" * 50)
    
    def check_detection_stability(self, garbage_type: str) -> bool:
        """Check if a detection is stable over time."""
        current_time = time.time()
        
        # If new type or detection interrupted
        if garbage_type != self.current_detection or (
            current_time - self.detection_lost_time > self.config.DETECTION_RESET_TIME
            and self.detection_lost_time > 0
        ):
            # Reset detection state
            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        
        # If stability threshold reached
        if (current_time - self.detection_start_time >= self.config.STABILITY_THRESHOLD
            and not self.stable_detection):
            self.stable_detection = True
            return True
            
        return self.stable_detection
    
    def can_count_new_garbage(self, garbage_type: str) -> bool:
        """Determine if a new garbage item can be counted."""
        current_time = time.time()
        
        # Check stability
        if not self.check_detection_stability(garbage_type):
            return False
        
        # If new type, reset lock
        if garbage_type != self.last_detected_type:
            self.is_counting_locked = False
            self.last_detected_type = garbage_type
        
        # Check cooldown period
        if self.is_counting_locked:
            if current_time - self.last_count_time >= self.config.COUNT_COOLDOWN:
                self.is_counting_locked = False
            else:
                return False
        
        return True
    
    def update_garbage_count(self, garbage_type: str) -> None:
        """Update the garbage count if conditions allow."""
        if not self.can_count_new_garbage(garbage_type):
            return
            
        self.garbage_count += 1
        self.detected_items.append({
            "count": self.garbage_count,
            "type": garbage_type,
            "quantity": 1,
            "status": "正确"
        })
        
        # Update state
        self.last_count_time = time.time()
        self.is_counting_locked = True
        self.last_detected_type = garbage_type
    
    def send_to_stm32(self, class_id: int, center_x: int, center_y: int) -> bool:
        """Send detection data to STM32 via serial."""
        if not self.config.ENABLE_SERIAL or self.state != SerialState.CONNECTED:
            return False
        
        # Map class 0 to prevent confusion with null byte
        mapped_class_id = self.zero_mapping if class_id == 0 else class_id
        
        # Scale coordinates to serial range (0-255)
        mapped_class_id = min(255, max(0, mapped_class_id))
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
            # Limit queue size
            if len(self.send_queue) >= 10:
                self.send_queue = self.send_queue[-9:]
                print("Warning: Queue full, dropping older items")
            
            self.send_queue.append({
                "class_id": mapped_class_id,
                "x": x_scaled,
                "y": y_scaled,
                "timestamp": time.time(),
                "orig_x": center_x,
                "orig_y": center_y,
                "orig_class": class_id,
                "retry": 0
            })
        
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False
        print("Cleaning up serial resources...")
        
        # Wait for thread to finish
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("Queue thread terminated")
        
        # Close port
        if self.port and self.port.is_open:
            self.port.close()
            print("Serial port closed")


class DetectionQueue:
    """Manages the queue of detections to be processed."""
    
    def __init__(self, config: Config, serial_manager: SerialManager):
        self.config = config
        self.serial_manager = serial_manager
        self.state = DetectionState.IDLE
        self.processing_queue = []
        self.is_processing = True
        self.queue_lock = threading.Lock()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("Detection processing thread started")
    
    def _process_queue(self) -> None:
        """Process detections in the queue."""
        while self.is_processing:
            detection_to_process = None
            
            # Get next detection
            with self.queue_lock:
                if self.processing_queue:
                    self.state = DetectionState.PROCESSING
                    detection_to_process = self.processing_queue.pop(0)
                else:
                    self.state = DetectionState.IDLE
            
            if detection_to_process:
                class_id, center_x, center_y, confidence, display_text, area = detection_to_process
                
                print(f"\nSending detection: {display_text}")
                print(f"Confidence: {confidence:.2%}")
                print(f"Center position: ({center_x}, {center_y})")
                print(f"Target area: {area} pixels^2")
                print("-" * 30)
                
                # Send to serial and update count
                self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                self.serial_manager.update_garbage_count(display_text)
                
                # Wait according to interval
                time.sleep(self.config.SEND_INTERVAL)
            else:
                # Sleep briefly if queue is empty
                time.sleep(0.1)
    
    def add_detection(self, detection: Dict) -> None:
        """Add a detection to the processing queue."""
        if not self._should_send_detection(
            detection["class_id"], 
            detection["center_x"], 
            detection["center_y"], 
            detection["confidence"]
        ):
            return
            
        # Get display text
        waste_classifier = WasteClassifier()
        category_id, description = waste_classifier.get_category_info(detection["class_id"])
        display_text = f"{category_id}({description})"
        
        # Add to queue
        with self.queue_lock:
            self.processing_queue.append((
                detection["class_id"],
                detection["center_x"],
                detection["center_y"],
                detection["confidence"],
                display_text,
                detection["area"]
            ))
            
            if self.config.DEBUG_WINDOW:
                print(f"Added to queue: {display_text}, area: {detection['area']}")
    
    def _should_send_detection(self, class_id: int, center_x: int, center_y: int, 
                               confidence: float) -> bool:
        """Determine if a detection should be sent."""
        # Basic confidence check
        return confidence >= self.config.CONF_THRESHOLD
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_processing = False
        print("Cleaning up detection queue...")
        
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
            print("Processing thread terminated")


class YOLODetector:
    """Handles object detection using YOLO."""
    
    def __init__(self, config: Config, detection_queue: DetectionQueue):
        self.config = config
        self.detection_queue = detection_queue
        self.state = DetectionState.IDLE
        
        # Setup model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config.MODEL_PATH)
        
        # Define class colors
        self.colors = {
            0: (86, 180, 233),   # 厨余垃圾 - 蓝色
            1: (230, 159, 0),    # 可回收垃圾 - 橙色
            2: (240, 39, 32),    # 有害垃圾 - 红色
            3: (0, 158, 115)     # 其他垃圾 - 绿色
        }
        
        # Detection throttling
        self.last_detection_time = 0
        self.last_detection_dict = {}
    
    def _calculate_area(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate the area of a detection box."""
        return abs((x2 - x1) * (y2 - y1))
    
    def _should_send_detection(self, class_id: int, center_x: int, center_y: int, 
                              confidence: float) -> bool:
        """Determine if a detection should be sent to processing."""
        current_time = time.time()
        
        # Check confidence
        if confidence < self.config.CONF_THRESHOLD:
            return False
        
        # Check time interval
        if current_time - self.last_detection_time < self.config.DETECTION_INTERVAL:
            # If new class, send it
            if class_id not in self.last_detection_dict:
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time
                }
                return True
            
            # Check position change
            last_pos = self.last_detection_dict[class_id]["position"]
            if (abs(center_x - last_pos[0]) > self.config.MIN_POSITION_CHANGE or
                abs(center_y - last_pos[1]) > self.config.MIN_POSITION_CHANGE):
                # Position changed significantly
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time
                }
                return True
            
            # Not enough change
            return False
        
        # Time interval sufficient, update state and send
        self.last_detection_time = current_time
        self.last_detection_dict[class_id] = {
            "position": (center_x, center_y),
            "time": current_time
        }
        return True
    
    def _visualize_detection(self, frame: np.ndarray, detection: Dict) -> None:
        """Add visualization elements for a detection."""
        if not self.config.DEBUG_WINDOW:
            return
        
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        center_x, center_y = detection["center_x"], detection["center_y"]
        class_id, confidence = detection["class_id"], detection["confidence"]
        area = detection["area"]
        
        # Draw bounding box
        color = self.colors.get(class_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Add label
        waste_classifier = WasteClassifier()
        category_id, description = waste_classifier.get_category_info(class_id)
        display_text = f"{category_id}({description})"
        
        label = f"{display_text} {confidence:.2f} A:{area}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects in a frame."""
        self.state = DetectionState.DETECTING
        results = self.model(frame, conf=self.config.CONF_THRESHOLD)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Collect all detections
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                area = self._calculate_area(x1, y1, x2, y2)
                
                detection = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": center_x, "center_y": center_y,
                    "confidence": confidence, "class_id": class_id,
                    "area": area
                }
                
                detections.append(detection)
                self._visualize_detection(frame, detection)
            
            # Sort by area, largest first
            detections.sort(key=lambda x: x["area"], reverse=True)
            
            # Process detections
            for detection in detections:
                if self._should_send_detection(
                    detection["class_id"],
                    detection["center_x"],
                    detection["center_y"],
                    detection["confidence"]
                ):
                    self.detection_queue.add_detection(detection)
                elif self.config.DEBUG_WINDOW:
                    waste_classifier = WasteClassifier()
                    category_id, description = waste_classifier.get_category_info(detection["class_id"])
                    display_text = f"{category_id}({description})"
                    print(f"Detected (not sending): {display_text}, area: {detection['area']}")
        
        self.state = DetectionState.IDLE
        return frame


class WasteClassificationSystem:
    """Main system for waste classification."""
    
    def __init__(self):
        self.config = Config()
        self.state = DetectionState.IDLE
        self.serial_manager = SerialManager(self.config)
        self.detection_queue = DetectionQueue(self.config, self.serial_manager)
        self.detector = YOLODetector(self.config, self.detection_queue)
        self.camera = None
    
    def initialize(self) -> bool:
        """Initialize the system."""
        # Setup GPU
        use_gpu, device_info = setup_gpu()
        print("\nDevice information:")
        print(device_info)
        print("-" * 30)
        
        # Find camera
        self.camera = find_camera()
        if not self.camera:
            self.state = DetectionState.ERROR
            return False
            
        # Print system info
        print("\nSystem started:")
        print("- Camera ready")
        print(f"- Debug window: {'Enabled' if self.config.DEBUG_WINDOW else 'Disabled'}")
        print(f"- Serial output: {'Enabled' if self.config.ENABLE_SERIAL else 'Disabled'}")
        print("- Press 'q' to exit")
        print("-" * 30)
        
        return True
    
    def run(self) -> None:
        """Run the main detection loop."""
        if not self.camera:
            print("Error: Camera not initialized")
            return
            
        try:
            while True:
                self.state = DetectionState.DETECTING
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Error: Cannot read from camera")
                    self.state = DetectionState.ERROR
                    break
                    
                # Process frame
                frame = self.detector.detect(frame)
                
                # Display debug window if enabled
                if self.config.DEBUG_WINDOW:
                    cv2.imshow("YOLO_detect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nProgram exited normally")
                        break
                        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, exiting")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up system resources."""
        print("Cleaning up resources...")
        
        # Clean up in reverse order of initialization
        if hasattr(self, "detection_queue"):
            self.detection_queue.cleanup()
            
        if hasattr(self, "serial_manager"):
            self.serial_manager.cleanup()
            
        if self.camera:
            self.camera.release()
            
        if self.config.DEBUG_WINDOW:
            cv2.destroyAllWindows()


def create_detector(model_path: str) -> YOLODetector:
    """Create a YOLODetector instance."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    file_extension = os.path.splitext(model_path)[1].lower()
    if file_extension != ".pt":
        raise ValueError(f"Unsupported model format: {file_extension}, only .pt supported")
        
    print(f"Using PyTorch model: {model_path}")
    
    # Create system
    system = WasteClassificationSystem()
    system.config.MODEL_PATH = model_path
    
    return system.detector


def main():
    """Main function."""
    system = WasteClassificationSystem()
    
    if system.initialize():
        system.run()


if __name__ == "__main__":
    main()
