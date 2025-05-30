import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
# use transitions lib
from transitions import Machine
# Third party function and class 
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# ============================================================
# Global config Variables(high pri) / Global Configuration Variables (High Priority)
# ============================================================
# Default configuration values
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9
model_path = "yolov12n_e300.pt"
STM32_PORT = "/dev/ttyUSB0" #choose any serial you want 
STM32_BAUD = 115200

# ============================================================
# Default configuration(low pri) / Default Configuration Parameters (Low Priority)
# ============================================================

@dataclass
class Config:
    """Centralized configuration for the application"""
    # Default offset
    class_id_offset: int = 1
    # Debug and operation flags
    debug_window: bool = True
    enable_serial: bool = True
    conf_threshold: float = 0.90
    
    # Model configuration
    model_path: str = "yolov12n_e300.pt"
    
    # Serial configuration
    #   (raspberrypi)：
    # name     | TX     | RX
    # ttyS0    | GPIO14 | GPIO15
    # ttyAMA2  | GPIO0  | GPIO1
    # ttyAMA3  | GPIO4  | GPIO5
    # ttyAMA4  | GPIO8  | GPIO9
    # ttyAMA5  | GPIO12 | GPIO13
    
    stm32_port: str = "/dev/ttyUSB0" 
    stm32_baud: int = 115200
    
    # Camera configuration
    camera_width: int = 1280
    camera_height: int = 720
    #Can be enable option
    crop_points: bool = False #List[Tuple[int, int]] = field(default_factory=lambda: [(465, 0), (1146, 720)])
    
    # Serial protocol configuration
    serial_header1: int = 0x2C
    serial_header2: int = 0x12
    serial_footer: int = 0x5B
    
    # Processing configuration
    min_position_change: int = 20
    send_interval: float = 0.0  # No delay between sending detections

# ============================================================
# Event System / Event System
# ============================================================

class DetectionState(Enum):
    """
    Detection system state enumeration
    IDLE: Idle state, waiting for new frame input
    DETECTING: Currently performing object detection
    PROCESSING: Currently processing detection results
    SENDING: Currently sending detection data to downstream devices
    ERROR: System error state
    """
    IDLE = auto()        # Idle state, waiting for new frames
    DETECTING = auto()   # Currently performing object detection
    PROCESSING = auto()  # Currently processing detection results
    SENDING = auto()     # Currently sending detection data
    ERROR = auto()       # Error state

class DetectionEvent(Enum):
    """
    Event enumeration that triggers state transitions
    FRAME_RECEIVED: Received new video frame
    DETECTION_COMPLETED: Completed object detection
    SEND_DETECTION: Send detection results
    DETECTION_SENT: Detection results have been sent
    ERROR_OCCURRED: Error occurred
    RESET: Reset system
    """
    FRAME_RECEIVED = auto()     # Received new video frame
    DETECTION_COMPLETED = auto() # Completed object detection
    SEND_DETECTION = auto()     # Event to send detection results
    DETECTION_SENT = auto()     # Detection results have been sent
    ERROR_OCCURRED = auto()     # Error occurred
    RESET = auto()              # Reset system

class EventBus:
    """
    Event bus class: Implements publish-subscribe pattern for decoupled communication between system components
    - Allows different components to subscribe to specific types of events
    - When events occur, notifies all components that subscribed to that event
    """
    def __init__(self):
        # Store mapping from event types to subscriber callback functions
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """
        Subscribe to specific type of event
        @param event_type: Event type, usually DetectionEvent enum value
        @param callback: Callback function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, *args, **kwargs):
        """
        Publish event to all subscribers
        @param event_type: Event type to publish
        @param args, kwargs: Parameters to pass to subscriber callback functions
        """
        if event_type in self.subscribers:
            # Call all callback functions that subscribed to this event
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)

# ============================================================
# Data Models / Data Models
# ============================================================

@dataclass
class Detection:
    """Data class for a single detection"""
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    confidence: float
    display_text: str
    area: int
    direction: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

# ============================================================
# Serial Communication Service / Serial Communication Service
# ============================================================

class SerialService:
    """Handles serial communication with the STM32 microcontroller"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.port = None
        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.is_running = True
        self.last_send_time = 0
        
        # For garbage type mapping
        waste_classifier = WasteClassifier()
        self.zero_mapping = max(waste_classifier.class_names.keys()) + 1
        print(f"Class 0 will be mapped to: {self.zero_mapping}")
        print(f"Overall offset: +{self.config.class_id_offset}")
        # Initialize serial port if enabled
        if self.config.enable_serial:
            self._initialize_serial()
            
        # Subscribe to detection events
        self.event_bus.subscribe(DetectionEvent.SEND_DETECTION, self.enqueue_detection)
    
    def _initialize_serial(self):
        """Initialize the serial port"""
        try:
            self.port = serial.Serial(
                self.config.stm32_port,
                self.config.stm32_baud,
                timeout=0.1,
                write_timeout=0.1
            )
            print(f"STM32 serial port initialized: {self.config.stm32_port}")
            
            # Start queue processing thread
            self.queue_thread = threading.Thread(target=self._process_queue)
            self.queue_thread.daemon = True
            self.queue_thread.start()
            print("Serial send queue processing thread started")
        except Exception as e:
            print(f"STM32 serial port initialization failed: {str(e)}")
            self.port = None
    
    def enqueue_detection(self, detection):
        """Add a detection to the send queue"""
        if not self.port or not self.is_running:
            return False
        # Map class ID 0 to a special value
        if detection.class_id == 0:
            mapped_class_id = self.zero_mapping + self.config.class_id_offset
        else:
            mapped_class_id = detection.class_id + self.config.class_id_offset
        
        # Ensure class_id is within valid range
        mapped_class_id = min(255, max(0, mapped_class_id))
        
        x_low = detection.center_x & 0xFF
        x_high = (detection.center_x >> 8) & 0xFF
        y_low = detection.center_y & 0xFF
        y_high = (detection.center_y >> 8) & 0xFF
        
        with self.queue_lock:
            # Limit queue size to avoid memory issues
            if len(self.send_queue) >= 10:
                self.send_queue = self.send_queue[-9:]
                print("Warning: Send queue is full, dropping old data")
            # Add data to queue
            self.send_queue.append({
                "class_id": mapped_class_id,
                "x_low": x_low,
                "x_high": x_high,
                "y_low": y_low,
                "y_high": y_high,
                "timestamp": time.time(),
                "orig_x": detection.center_x,
                "orig_y": detection.center_y,
                "orig_class": detection.class_id,
                "retry": 0,
                "direction": detection.direction,
            })
        
        return True
    
    def _process_queue(self):
        """Process the send queue continuously"""
        while self.is_running:
            try:
                self._send_next_item()
            except Exception as e:
                print(f"Queue processing exception: {str(e)}")
            # Short sleep
            time.sleep(0.01)
    
    def _send_next_item(self):
        """Send the next item in the queue"""
        # Check serial port status
        if not self.port:
            return
        
        if not self.port.is_open:
            try:
                print("Attempting to reopen serial port...")
                self.port.open()
                print("Serial port reopened successfully")
            except Exception as e:
                print(f"Failed to reopen serial port: {str(e)}")
                return
        
        # Get next item from queue
        data_to_send = None
        with self.queue_lock:
            if self.send_queue:
                data_to_send = self.send_queue.pop(0)
        
        if not data_to_send:
            return
        
        try:
            # Construct data packet
            data = bytes([
                self.config.serial_header1,
                self.config.serial_header2,
                data_to_send["class_id"],
                data_to_send["x_high"],
                data_to_send["x_low"],
                data_to_send["y_high"],
                data_to_send["y_low"],
                data_to_send["direction"],
                self.config.serial_footer,
            ])
            
            # Send data
            bytes_written = self.port.write(data)
            self.port.flush()
            self.last_send_time = time.time()
            
            # Debug output
            if self.config.debug_window:
                print("\n----- Serial Send Detailed Data [DEBUG] -----")
                print(f"Hex data: {' '.join([f'0x{b:02X}' for b in data])}")
                
                print("Raw packet structure:")
                print(f"  [0] 0x{data[0]:02X} - Header1")
                print(f"  [1] 0x{data[1]:02X} - Header2")
                print(f"  [2] 0x{data[2]:02X} - Class ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})")
                print(f"  [3] 0x{data[3]:02X} - X coordinate high 8 bits")
                print(f"  [4] 0x{data[4]:02X} - X coordinate low 8 bits")
                print(f"  [5] 0x{data[5]:02X} - Y coordinate high 8 bits")
                print(f"  [6] 0x{data[6]:02X} - Y coordinate low 8 bits")
                print(f"  [7] 0x{data[7]:02X} - Direction (1: w>h, 2: w<h)")
                print(f"  [8] 0x{data[8]:02X} - Footer")
                print(f"Total packet length: {len(data)} bytes, actually written: {bytes_written} bytes")
                print(f"Original class ID: {data_to_send['orig_class']} (decimal) -> {data_to_send['class_id']} (sent value)")
                print(f"Original X coordinate: {data_to_send['orig_x']} -> Split: low 8 bits=0x{data_to_send['x_low']:02X}, high 8 bits=0x{data_to_send['x_high']:02X}")
                print(f"Original Y coordinate: {data_to_send['orig_y']} -> Split: low 8 bits=0x{data_to_send['y_low']:02X}, high 8 bits=0x{data_to_send['y_high']:02X}")
                print(f"Direction: {data_to_send['direction']}")
                print(f"Data queue waiting time: {time.time() - data_to_send['timestamp']:.3f} seconds")
                print("-" * 50)
            
            # Notify that detection was sent
            self.event_bus.publish(DetectionEvent.DETECTION_SENT, data_to_send)
            
        except serial.SerialTimeoutException:
            print("Serial write timeout, device may not be responding")
            # Put back in queue for retry
            with self.queue_lock:
                self.send_queue.insert(0, data_to_send)
        
        except Exception as e:
            print(f"Serial send error: {str(e)}")
            # Retry sending data
            with self.queue_lock:
                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # Maximum 3 retries
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send)
                    print(f"Data will retry sending, attempt #{retry_count}")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("Cleaning up serial resources...")
        
        # Wait for queue thread to end
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("Queue processing thread terminated")
        
        # Close serial port
        if self.port and self.port.is_open:
            self.port.close()
            print("Serial port closed")


# ============================================================
# Detection Service / Detection Service - Refactored using transitions library
# ============================================================

class DetectionService:
    """Handles object detection using YOLO with transitions state machine"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        
        # Initialize state attributes
        self.state = DetectionState.IDLE
        
        # Colors for visualization
        self.colors = {
            0: (86, 180, 233),   # Kitchen waste - Blue
            1: (230, 159, 0),    # Recyclable waste - Orange
            2: (240, 39, 32),    # Hazardous waste - Red
            3: (0, 158, 115),    # Other waste - Green
        }
        
        # Detection state
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.is_processing = False
        self.last_detection_dict = {}
        self.waste_classifier = WasteClassifier()
        
        # Setup transitions state machine
        self._setup_state_machine()
        
        # Load YOLO model
        self._load_model()
        
        # Start processing thread
        self.start_processing_thread()
    
    def _setup_state_machine(self):
        """
        Setup state machine using transitions library
        """
        # Define state transition rules
        transitions = [
            # IDLE state -> DETECTING state: when receiving new video frame
            {'trigger': 'frame_received', 'source': DetectionState.IDLE, 'dest': DetectionState.DETECTING},
            # Allow transition from any state back to DETECTING state for enhanced fault tolerance
            {'trigger': 'frame_received', 'source': '*', 'dest': DetectionState.DETECTING},
            
            # DETECTING state -> PROCESSING state: when detection is completed
            {'trigger': 'detection_completed', 'source': DetectionState.DETECTING, 'dest': DetectionState.PROCESSING},
            
            # PROCESSING state -> SENDING state: when need to send detection results
            {'trigger': 'send_detection', 'source': DetectionState.PROCESSING, 'dest': DetectionState.SENDING},
            # Allow direct transition from IDLE state to SENDING state for added flexibility
            {'trigger': 'send_detection', 'source': DetectionState.IDLE, 'dest': DetectionState.SENDING},
            
            # SENDING state -> IDLE state: when detection results have been sent
            {'trigger': 'detection_sent', 'source': DetectionState.SENDING, 'dest': DetectionState.IDLE},
            
            # Error handling: any state can transition to ERROR state when error occurs
            {'trigger': 'error_occurred', 'source': '*', 'dest': DetectionState.ERROR},
            
            # Reset from ERROR state: return to IDLE state through RESET event
            {'trigger': 'reset', 'source': DetectionState.ERROR, 'dest': DetectionState.IDLE}
        ]
        
        # Initialize transitions state machine, set ignore_invalid_triggers=True for enhanced fault tolerance
        self.machine = Machine(
            model=self, 
            states=list(DetectionState), 
            transitions=transitions, 
            initial=DetectionState.IDLE, 
            send_event=True,
            ignore_invalid_triggers=True  # Ignore invalid triggers to avoid throwing exceptions
        )
        
        # Set up mapping between state machine events and EventBus
        self.event_map = {
            DetectionEvent.FRAME_RECEIVED: self.frame_received,
            DetectionEvent.DETECTION_COMPLETED: self.detection_completed,
            DetectionEvent.SEND_DETECTION: self.send_detection,
            DetectionEvent.DETECTION_SENT: self.detection_sent,
            DetectionEvent.ERROR_OCCURRED: self.error_occurred,
            DetectionEvent.RESET: self.reset
        }
        
        # Subscribe to EventBus events, using modified callbacks to pass parameters
        for event, handler in self.event_map.items():
            self.event_bus.subscribe(event, lambda e=None, handler=handler, *args, **kwargs: handler(e))
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(self.config.model_path)
            print(f"YOLO model loaded: {self.config.model_path}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            self.event_bus.publish(DetectionEvent.ERROR_OCCURRED, str(e))
    
    def start_processing_thread(self):
        """Start the detection processing thread"""
        self.is_processing = True
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("Detection sequential processing thread started")
    
    def detect(self, frame):
        """Detect objects in a frame"""
        try:
            # Use try/except to wrap state machine calls, ensuring detection functionality remains available even if state machine fails
            try:
                # Update state machine state
                self.frame_received()
            except Exception as state_error:
                print(f"State transition warning: {str(state_error)}")
                # If in ERROR state, try to reset state machine
                if self.state == DetectionState.ERROR:
                    try:
                        self.reset()
                        print("State machine reset from ERROR state")
                        self.frame_received()  # Try to trigger frame received event again
                    except Exception:
                        pass  # Ignore possible reset errors, continue with detection functionality
            
            # Perform detection
            results = self.model(frame, conf=self.config.conf_threshold)
            detections = []
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Process detected objects
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Get category information
                    category_id, description = self.waste_classifier.get_category_info(class_id)
                    display_text = f"{category_id}({description})"
                    
                    # Calculate area and direction
                    area = self._calculate_area(x1, y1, x2, y2)
                    w, h = x2 - x1, y2 - y1
                    direction = self._determine_direction(w, h)
                    
                    # Create detection object
                    detection = Detection(
                        class_id=class_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=center_x, center_y=center_y,
                        confidence=confidence,
                        display_text=display_text,
                        area=area,
                        direction=direction
                    )
                    
                    # Add to detection list
                    detections.append(detection)
                    
                    # If debug window is enabled, visualize results
                    if self.config.debug_window:
                        color = self.colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                        
                        label = f"{display_text} {confidence:.2f} A:{area} D:{direction}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                        )
                        cv2.rectangle(
                            frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                        )
                
                # Sort by area (larger objects first)
                detections.sort(key=lambda x: x.area, reverse=True)
                
                # Add to processing queue
                with self.queue_lock:
                    # Clear current queue
                    self.processing_queue = []
                    
                    # Add new detection objects
                    for detection in detections:
                        # Always process detection objects (time-based filtering removed)
                        self.processing_queue.append(detection)
                        
                        if self.config.debug_window:
                            print(f"Added to queue: {detection.display_text}, Area: {detection.area}, Direction: {detection.direction}")
            
            # Use try/except to wrap state machine calls
            try:
                # Update state machine state
                self.detection_completed()
            except Exception as state_error:
                print(f"State transition warning: {str(state_error)}")
            
            return frame
            
        except Exception as e:
            print(f"Frame processing exception: {str(e)}")
            try:
                self.error_occurred()
            except Exception:
                print("Cannot transition to ERROR state")
            return frame
    
    def _calculate_area(self, x1, y1, x2, y2):
        """Calculate area of detection box"""
        return abs((x2 - x1) * (y2 - y1))
    
    def _determine_direction(self, w, h):
        """Determine direction based on width and height"""
        return 1 if w > h else 2
    
    def _should_process_detection(self, detection):
        """Determine if a detection should be processed
        Note: As requested, we'll always return True to process all detections
        """
        return True
    
    def _process_queue(self):
        """Process detection queue"""
        while self.is_processing:
            try:
                detection_to_process = None
                
                with self.queue_lock:
                    if self.processing_queue:
                        detection_to_process = self.processing_queue.pop(0)
                
                if detection_to_process:
                    # Output detailed information
                    print(f"\nSending detection: {detection_to_process.display_text}")
                    print(f"Confidence: {detection_to_process.confidence:.2%}")
                    print(f"Center position: ({detection_to_process.center_x}, {detection_to_process.center_y})")
                    print(f"Object area: {detection_to_process.area} pixels^2")
                    print(f"Direction: {detection_to_process.direction}")
                    print("-" * 30)
                    
                    # Only trigger state machine changes when in processing state, publish events directly in other states
                    # This avoids state machine errors while maintaining system functionality
                    try:
                        # Try to trigger state machine event
                        if self.state == DetectionState.PROCESSING:
                            self.send_detection()
                    except Exception as e:
                        print(f"State transition warning (non-fatal): {str(e)}")
                    
                    # Regardless of state machine state, ensure detection data is sent to serial service
                    self.event_bus.publish(DetectionEvent.SEND_DETECTION, detection_to_process)
                    
                    # Wait for specified time interval (default is 0)
                    time.sleep(self.config.send_interval)
                else:
                    # Queue is empty, sleep briefly to avoid high CPU usage
                    time.sleep(0.1)
            except Exception as e:
                print(f"Queue processing exception: {str(e)}")
                # Sleep briefly to avoid high CPU usage and prevent infinite error loops
                time.sleep(0.1)
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up DetectionService resources...")
        self.is_processing = False
        
        if hasattr(self, "process_thread") and self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0)
                print("Detection processing thread terminated")
            except Exception as e:
                print(f"Error terminating processing thread: {str(e)}")


# ============================================================
# Statistics Manager / Statistics Manager
# ============================================================

class StatisticsManager:
    """Manages detection statistics"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.garbage_count = 0
        self.detected_items = []
        
        # Subscribe to detection events
        self.event_bus.subscribe(DetectionEvent.SEND_DETECTION, self.update_statistics)
    
    def update_statistics(self, detection):
        """Update statistics for a detected item"""
        self.garbage_count += 1
        self.detected_items.append({
            "count": self.garbage_count,
            "type": detection.display_text,
            "quantity": 1,
            "status": "Correct",
        })
    
    def get_statistics(self):
        """Get current statistics"""
        return {
            "total_count": self.garbage_count,
            "items": self.detected_items
        }


# ============================================================
# Application Class / Integration Layer
# ============================================================

class WasteDetectionApp:
    """Main application for waste detection and classification"""
    
    def __init__(self, config=None):
        """Initialize the application"""
        # Load global configuration
        self.config = config or Config()
        
        # Create event bus for communication between components
        self.event_bus = EventBus()
        
        # Set up GPU
        self._setup_gpu()
        
        # Initialize components
        self.detection_service = DetectionService(self.config, self.event_bus)
        self.serial_service = SerialService(self.config, self.event_bus)
        self.statistics_manager = StatisticsManager(self.event_bus)
    
    def _setup_gpu(self):
        """Set up GPU if available"""
        use_gpu, device_info = setup_gpu()
        print("\nDevice Information:")
        print(device_info)
        print("-" * 30)
    
    def run(self):
        """Run the application"""
        # Initialize camera
        cap = find_camera(self.config.camera_width,self.config.camera_height)
        if not cap:
            print("Camera not found")
            return
        print("\nSystem Startup:")
        print("- Camera ready")
        print(f"- Debug window: {'Enabled' if self.config.debug_window else 'Disabled'}")
        print(f"- Serial output: {'Enabled' if self.config.enable_serial else 'Disabled'}")
        print("- Press 'q' key to exit program")
        print("-" * 30)
        
        try:
            while True:
                # Read
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read camera feed")
                    break
                
                # Crop
                if self.config.crop_points:
                    frame = crop_frame(frame, points=self.config.crop_points)
                
                # Process
                frame = self.detection_service.detect(frame)
                
                # Display
                if self.config.debug_window:
                    window_name = "YOLO_detect"
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nProgram exited normally")
                        break
                        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, program exiting")
            
        finally:
            # Clean up resources
            self.cleanup(cap)
    
    def cleanup(self, cap):
        """Clean up all resources"""
        # Clean up components
        self.detection_service.cleanup()
        self.serial_service.cleanup()
        
        # Release camera
        cap.release()
        
        # Close windows
        if self.config.debug_window:
            cv2.destroyAllWindows()


# ============================================================
# Main Function / Main Function
# ============================================================

def main():
    """Main function"""
    # Create application with default config
    app = WasteDetectionApp(Config(
        debug_window=DEBUG_WINDOW,
        enable_serial=ENABLE_SERIAL,
        conf_threshold=CONF_THRESHOLD,
        model_path=os.path.join(get_script_directory(), model_path),
        stm32_port=STM32_PORT,
        stm32_baud=STM32_BAUD
    ))
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
