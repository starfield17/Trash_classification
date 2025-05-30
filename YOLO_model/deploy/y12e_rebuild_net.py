import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import os
import socket  # Added for network debugging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
# Third-party functions and classes
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# ============================================================
# Global config Variables (high priority)
# ============================================================
# Default configuration values
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9
model_path = "yolov12n_e300.pt"
STM32_PORT = "/dev/ttyUSB0"  # Choose any serial port you want
STM32_BAUD = 115200
ENABLE_DEBUG_PORT = True     # Enable network debug port
DEBUG_PORT = 8964            # Network port for debugging

# ============================================================
# Default configuration (low priority)
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
    # Can be an enable option
    crop_points: bool = False # List[Tuple[int, int]] = field(default_factory=lambda: [(465, 0), (1146, 720)])
    
    # Serial protocol configuration
    serial_header1: int = 0x2C
    serial_header2: int = 0x12
    serial_footer: int = 0x5B
    
    # Processing configuration
    min_position_change: int = 20
    send_interval: float = 0.0  # No delay between sending detections
    
    # Network debug configuration
    enable_debug_port: bool = True  # Enable network debug port
    debug_port: int = 8964           # Port for debug commands

# ============================================================
# Event System
# ============================================================

class DetectionState(Enum):
    """
    Enumeration of states for the detection system.
    IDLE: Idle state, waiting for new frame input.
    DETECTING: Performing object detection.
    PROCESSING: Processing detection results.
    SENDING: Sending detection data to downstream devices.
    ERROR: System error state.
    """
    IDLE = auto()        # Idle state, waiting for a new frame
    DETECTING = auto()   # Performing object detection
    PROCESSING = auto()  # Processing detection results
    SENDING = auto()     # Sending detection data
    ERROR = auto()       # Error state

class DetectionEvent(Enum):
    """
    Enumeration of events that trigger state transitions.
    FRAME_RECEIVED: A new video frame has been received.
    DETECTION_COMPLETED: Object detection has finished.
    SEND_DETECTION: Send detection results.
    DETECTION_SENT: Detection results have been sent.
    ERROR_OCCURRED: An error has occurred.
    RESET: Reset the system.
    """
    FRAME_RECEIVED = auto()     # Received a new video frame
    DETECTION_COMPLETED = auto() # Object detection completed
    SEND_DETECTION = auto()     # Event to send detection results
    DETECTION_SENT = auto()     # Detection results have been sent
    ERROR_OCCURRED = auto()     # An error occurred
    RESET = auto()              # Reset the system

class EventBus:
    """
    Event bus class: Implements a publish-subscribe pattern for decoupled communication
    between system components.
    - Allows different components to subscribe to specific types of events.
    - Notifies all subscribed components when an event occurs.
    """
    def __init__(self):
        # Stores a mapping from event types to subscriber callback functions
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """
        Subscribe to a specific type of event.
        @param event_type: The type of event, usually a DetectionEvent enum value.
        @param callback: The callback function to be called when the event occurs.
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, *args, **kwargs):
        """
        Publish an event to all subscribers.
        @param event_type: The type of event to publish.
        @param args, kwargs: Arguments to pass to the subscriber callback functions.
        """
        if event_type in self.subscribers:
            # Call all callback functions subscribed to this event
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)


# ============================================================
# State Machine
# ============================================================

class StateMachine:
    """
    Generic state machine implementation.
    - Manages system state transitions.
    - Events trigger state transitions.
    - Executes callback functions during state transitions.
    """
    
    def __init__(self, initial_state):
        """
        Initialize the state machine.
        @param initial_state: The initial state, usually a DetectionState enum value.
        """
        self.state = initial_state           # Current state
        self.transitions = {}                # Stores state transition rules
        self.callbacks = {}                  # Stores callback functions for transitions
        
    def add_transition(self, from_state, event, to_state, callback=None):
        """
        Add a state transition rule.
        @param from_state: The starting state.
        @param event: The event that triggers the transition.
        @param to_state: The target state.
        @param callback: Optional callback function to execute during the state transition.
        """
        # If the starting state is not in the transition table, add it
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        # Set the transition rule from the starting state via the event to the target state
        self.transitions[from_state][event] = to_state
        
        # If a callback function is provided, store it
        if callback:
            # Store the callback function for the specific (state, event) pair
            if (from_state, event) not in self.callbacks:
                self.callbacks[(from_state, event)] = []
            self.callbacks[(from_state, event)].append(callback)
    
    def trigger(self, event, *args, **kwargs):
        """
        Trigger an event, attempting to execute a state transition.
        @param event: The event to trigger.
        @param args, kwargs: Arguments to pass to the callback functions.
        @return: True if the state transition was successful, False otherwise.
        """
        # Check if there is a transition rule for the current state and event
        if self.state in self.transitions and event in self.transitions[self.state]:
            # Execute all callbacks associated with this state and event
            if (self.state, event) in self.callbacks:
                for callback in self.callbacks[(self.state, event)]:
                    callback(*args, **kwargs)
            
            # Perform the state transition
            old_state = self.state
            self.state = self.transitions[self.state][event]
            # print(f"State transition: {old_state} -> {self.state} on event {event}") # Optional: for debugging
            return True
        # print(f"No transition for state {self.state} on event {event}") # Optional: for debugging
        return False
    
    def get_state(self):
        """
        Get the current state.
        @return: The current state value.
        """
        return self.state


# ============================================================
# Data Models
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
    direction: int # 1 for horizontal (width > height), 2 for vertical (height >= width)
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


# ============================================================
# Serial Communication Service
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
        
        # Start debug network listener if enabled
        if self.config.enable_debug_port:
            self._start_debug_listener()
    
    def _initialize_serial(self):
        """Initialize the serial port"""
        try:
            self.port = serial.Serial(
                self.config.stm32_port,
                self.config.stm32_baud,
                timeout=0.1,      # Read timeout
                write_timeout=0.1 # Write timeout
            )
            print(f"STM32 serial port initialized: {self.config.stm32_port}")
            
            # Start queue processing thread
            self.queue_thread = threading.Thread(target=self._process_queue)
            self.queue_thread.daemon = True
            self.queue_thread.start()
            print("Serial send queue processing thread started")
        except Exception as e:
            print(f"STM32 serial port initialization failed: {str(e)}")
            self.port = None # Ensure port is None if initialization fails
    
    def _start_debug_listener(self):
        """Start the network debug listener"""
        try:
            # Create a thread to listen for debug commands
            self.debug_thread = threading.Thread(target=self._debug_listener)
            self.debug_thread.daemon = True
            self.debug_thread.start()
            print(f"Debug network listener started on port: {self.config.debug_port}")
        except Exception as e:
            print(f"Failed to start debug listener: {str(e)}")
    
    def _debug_listener(self):
        """Listen for debug commands on the network port"""
        try:
            # Create a socket server
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow address reuse
            server_socket.bind(('0.0.0.0', self.config.debug_port))
            server_socket.listen(5) # Listen for up to 5 queued connections
            
            print(f"Debug server started, listening on port {self.config.debug_port}")
            
            while self.is_running:
                try:
                    # Accept connections
                    client_socket, addr = server_socket.accept()
                    print(f"Received debug connection from {addr}")
                    
                    # Receive data (up to 1024 bytes)
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    client_socket.close() # Close connection after receiving data
                    
                    # Process the command
                    self._process_debug_command(data)
                except socket.timeout: # Handle timeout if server_socket.accept() has a timeout
                    continue
                except Exception as e:
                    print(f"Error handling debug connection: {str(e)}")
        
        except Exception as e:
            print(f"Debug listener exception: {str(e)}")
        finally:
            try:
                server_socket.close() # Ensure socket is closed
            except:
                pass # Ignore errors if socket is already closed or invalid
    
    def _process_debug_command(self, command):
        """Process a debug command received from the network"""
        try:
            # Try to parse the command as an integer
            class_id = int(command.strip())
            
            # Validate class_id (assuming 0-3 are valid debug commands)
            if class_id < 0 or class_id > 3: # Example range, adjust as needed
                print(f"Warning: Received invalid class ID: {class_id}, valid range is 0-3")
                return
            
            print(f"Received debug command: Send class ID {class_id}")
            
            # Create a manual detection with fixed parameters
            # Fixed 1x1 box at (0,0) to indicate a manual debug message
            manual_detection = Detection(
                class_id=class_id,
                x1=0, y1=0, x2=1, y2=1,  # 1x1 box at origin
                center_x=0, center_y=0,  # Center at origin
                confidence=1.0,          # 100% confidence
                display_text=f"DEBUG_{class_id}",
                area=1,                  # Area = 1x1 = 1
                direction=1              # Direction = 1 (width > height)
            )
            
            # Directly enqueue for sending
            self.enqueue_debug_detection(manual_detection)
            
        except ValueError:
            print(f"Warning: Could not parse debug command: '{command}', expected integer 0-3")
        except Exception as e:
            print(f"Error processing debug command: {str(e)}")
    
    def enqueue_debug_detection(self, detection):
        """Add a debug detection to the send queue with special flags"""
        if not self.port or not self.is_running:
            # print("Serial port not available or service not running, cannot enqueue debug detection.")
            return False
            
        # Map class ID with offset
        if detection.class_id == 0:
            mapped_class_id = self.zero_mapping + self.config.class_id_offset
        else:
            mapped_class_id = detection.class_id + self.config.class_id_offset
        
        # Ensure class_id is within valid range (0-255 for byte)
        mapped_class_id = min(255, max(0, mapped_class_id))
        
        # For debug messages, use fixed coordinates (0,0)
        x_low = 0
        x_high = 0
        y_low = 0
        y_high = 0
        
        with self.queue_lock:
            # Add data to queue with debug flag
            self.send_queue.append({
                "class_id": mapped_class_id,
                "x_low": x_low,
                "x_high": x_high,
                "y_low": y_low,
                "y_high": y_high,
                "timestamp": time.time(),
                "orig_x": 0, # Original X for debug
                "orig_y": 0, # Original Y for debug
                "orig_class": detection.class_id, # Original class ID
                "retry": 0,
                "direction": detection.direction,
                "is_debug": True  # Flag to indicate this is a debug message
            })
            
            print("Debug message added to send queue")
        
        return True
    
    def enqueue_detection(self, detection):
        """Add a detection to the send queue"""
        if not self.port or not self.is_running:
            # print("Serial port not available or service not running, cannot enqueue detection.")
            return False
        # Map class ID 0 to a special value
        if detection.class_id == 0:
            mapped_class_id = self.zero_mapping + self.config.class_id_offset
        else:
            mapped_class_id = detection.class_id + self.config.class_id_offset
        
        # Ensure class_id is within valid range (0-255 for byte)
        mapped_class_id = min(255, max(0, mapped_class_id))
        
        # Split center coordinates into high and low bytes
        x_low = detection.center_x & 0xFF
        x_high = (detection.center_x >> 8) & 0xFF
        y_low = detection.center_y & 0xFF
        y_high = (detection.center_y >> 8) & 0xFF
        
        with self.queue_lock:
            # Limit queue size to avoid memory issues
            if len(self.send_queue) >= 10: # Max 10 items in queue
                self.send_queue = self.send_queue[-9:] # Keep the newest 9 items
                print("Warning: Send queue full, discarding oldest data.")
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
                "is_debug": False  # Regular detection
            })
        
        return True
    
    def _process_queue(self):
        """Process the send queue continuously"""
        while self.is_running:
            try:
                self._send_next_item()
            except Exception as e:
                print(f"Queue processing exception: {str(e)}")
            # Short sleep to prevent busy-waiting
            time.sleep(0.01) # Adjust sleep time as needed
    
    def _send_next_item(self):
        """Send the next item in the queue"""
        # Check serial port status
        if not self.port:
            # print("Serial port not initialized, cannot send.") # Can be noisy
            return
        
        if not self.port.is_open:
            try:
                print("Attempting to reopen serial port...")
                self.port.open()
                print("Serial port reopened successfully.")
            except Exception as e:
                print(f"Failed to reopen serial port: {str(e)}")
                return # Cannot send if port cannot be opened
        
        # Get next item from queue
        data_to_send = None
        with self.queue_lock:
            if self.send_queue:
                data_to_send = self.send_queue.pop(0) # Get the oldest item
        
        if not data_to_send:
            return # Queue is empty
        
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
                data_to_send["direction"], # Added direction byte
                self.config.serial_footer,
            ])
            
            # Send data
            bytes_written = self.port.write(data)
            self.port.flush() # Ensure all data is written
            self.last_send_time = time.time()
            
            # Debug output
            if self.config.debug_window: # Or a more specific serial debug flag
                print("\n----- Serial Send Detailed Data [DEBUG] -----")
                if data_to_send.get("is_debug", False):
                    print("*** Manual Debug Message ***")
                print(f"Hex data: {' '.join([f'0x{b:02X}' for b in data])}")
                
                print("Original packet structure:")
                print(f"  [0] 0x{data[0]:02X} - Frame Header 1")
                print(f"  [1] 0x{data[1]:02X} - Frame Header 2")
                print(f"  [2] 0x{data[2]:02X} - Class ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})")
                print(f"  [3] 0x{data[3]:02X} - X Coordinate High 8 bits")
                print(f"  [4] 0x{data[4]:02X} - X Coordinate Low 8 bits")
                print(f"  [5] 0x{data[5]:02X} - Y Coordinate High 8 bits")
                print(f"  [6] 0x{data[6]:02X} - Y Coordinate Low 8 bits")
                print(f"  [7] 0x{data[7]:02X} - Direction (1: w>h, 2: w<h)")
                print(f"  [8] 0x{data[8]:02X} - Frame Footer")
                print(f"Total packet length: {len(data)} bytes, actually written: {bytes_written} bytes")
                print(f"Original Class ID: {data_to_send['orig_class']} (decimal) -> {data_to_send['class_id']} (sent value)")
                print(f"Original X Coordinate: {data_to_send['orig_x']} -> Split: Low 8-bit=0x{data_to_send['x_low']:02X}, High 8-bit=0x{data_to_send['x_high']:02X}")
                print(f"Original Y Coordinate: {data_to_send['orig_y']} -> Split: Low 8-bit=0x{data_to_send['y_low']:02X}, High 8-bit=0x{data_to_send['y_high']:02X}")
                print(f"Direction: {data_to_send['direction']}")
                print(f"Data waited in queue for: {time.time() - data_to_send['timestamp']:.3f} seconds")
                print("-" * 50)
            
            # Notify that detection was sent
            self.event_bus.publish(DetectionEvent.DETECTION_SENT, data_to_send)
            
        except serial.SerialTimeoutException:
            print("Serial write timeout, device might not be responding.")
            # Put back in queue for retry
            with self.queue_lock:
                self.send_queue.insert(0, data_to_send) # Add back to the front
        
        except Exception as e:
            print(f"Serial send error: {str(e)}")
            # Retry sending data
            with self.queue_lock:
                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # Maximum 3 retries
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send) # Add back to the front
                    print(f"Data will be retried, attempt {retry_count}")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("Cleaning up serial resources...")
        
        # Wait for queue thread to end
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0) # Wait for 2 seconds
            print("Queue processing thread terminated.")
        
        # Wait for debug thread to end
        if hasattr(self, "debug_thread") and self.debug_thread.is_alive():
            # To stop the debug listener, you might need to make a dummy connection
            # or use a more sophisticated shutdown mechanism for the socket.
            # For now, we rely on self.is_running and join.
            try:
                # Create a dummy connection to unblock accept()
                # This is a common way to shut down a blocking socket server
                if self.config.enable_debug_port:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect(('127.0.0.1', self.config.debug_port))
            except Exception:
                pass # Ignore if connection fails (e.g., server already down)
            self.debug_thread.join(timeout=2.0)
            print("Debug listener thread terminated.")
        
        # Close serial port
        if self.port and self.port.is_open:
            self.port.close()
            print("Serial port closed.")


# ============================================================
# Detection Service
# ============================================================

class DetectionService:
    """Handles object detection using YOLO"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.state_machine = StateMachine(DetectionState.IDLE) # Initial state is IDLE
        self.waste_classifier = WasteClassifier() # For classifying waste types
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            0: (233, 180, 86),   # Kitchen Waste - Blue
            1: (0, 159, 230),    # Recyclable Waste - Orange
            2: (32, 39, 240),    # Hazardous Waste - Red
            3: (115, 158, 0),    # Other Waste - Green
        } # Ensure these colors match your classes
        
        # Detection state
        self.processing_queue = [] # Queue for detections to be processed sequentially
        self.queue_lock = threading.Lock()
        self.is_processing = False # Flag to control the processing thread
        self.last_detection_dict = {} # Stores last detection time for each class_id to avoid spam
        
        # Configure state machine
        self._setup_state_machine()
        
        # Load YOLO model
        self._load_model()
        
        # Start processing thread
        self.start_processing_thread()
    
    def _setup_state_machine(self):
        """
        Set up the state machine's transition rules.
        1. From IDLE state, transition to DETECTING state upon receiving a new video frame.
        2. After detection is complete in DETECTING state, transition to PROCESSING state.
        3. From PROCESSING state, when preparing to send data, transition to SENDING state.
        4. After data is sent, transition from SENDING state back to IDLE state, ready for the next frame.
        5. If an error occurs in any state, transition to ERROR state.
        6. From ERROR state, reset to IDLE state via a RESET event.
        """
        # IDLE state -> DETECTING state: When a new video frame is received
        # Starting point of the state machine, system starts detecting from idle upon receiving a new frame.
        self.state_machine.add_transition(
            DetectionState.IDLE,                # Starting state: Idle
            DetectionEvent.FRAME_RECEIVED,      # Triggering event: New frame received
            DetectionState.DETECTING            # Target state: Detecting
        )
        
        # DETECTING state -> PROCESSING state: When detection is complete
        # After object detection is finished, the system needs to process the results.
        self.state_machine.add_transition(
            DetectionState.DETECTING,           # Starting state: Detecting
            DetectionEvent.DETECTION_COMPLETED, # Triggering event: Detection complete
            DetectionState.PROCESSING           # Target state: Processing results
        )
        
        # PROCESSING state -> SENDING state: When detection results need to be sent
        # After processing results, the system prepares to send data to downstream devices (e.g., STM32).
        self.state_machine.add_transition(
            DetectionState.PROCESSING,          # Starting state: Processing results
            DetectionEvent.SEND_DETECTION,      # Triggering event: Send detection results
            DetectionState.SENDING              # Target state: Sending
        )
        
        # SENDING state -> IDLE state: When detection results have been sent
        # After sending, the system returns to idle, waiting for the next frame.
        self.state_machine.add_transition(
            DetectionState.SENDING,             # Starting state: Sending
            DetectionEvent.DETECTION_SENT,      # Triggering event: Data sent
            DetectionState.IDLE                 # Target state: Return to idle
        )
        
        # Error handling: Transition to ERROR state if an error occurs in any state
        # This is a global error handling mechanism to ensure the system enters a controllable error state during any exception.
        for state in DetectionState:
            if state != DetectionState.ERROR:   # For all states except ERROR
                self.state_machine.add_transition(
                    state,                       # Any starting state
                    DetectionEvent.ERROR_OCCURRED, # Triggering event: Error occurred
                    DetectionState.ERROR         # Target state: Error state
                )
        
        # Reset from ERROR state: Return to IDLE state via RESET event
        # Provides a mechanism to recover from an error state and restart the detection process.
        self.state_machine.add_transition(
            DetectionState.ERROR,               # Starting state: Error state
            DetectionEvent.RESET,               # Triggering event: Reset
            DetectionState.IDLE                 # Target state: Return to idle
        )
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(self.config.model_path)
            # self.model.to(self.device) # If model needs to be explicitly moved to device
            print(f"YOLO model loaded: {self.config.model_path} on device {self.device}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            self.event_bus.publish(DetectionEvent.ERROR_OCCURRED, str(e))
            # Potentially raise the exception or handle it to prevent app from continuing without a model
    
    def start_processing_thread(self):
        """Start the detection processing thread"""
        self.is_processing = True
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("Detection sequential processing thread started.")
    
    def detect(self, frame):
        """Detect objects in a frame"""
        try:
            # Update state machine
            if not self.state_machine.trigger(DetectionEvent.FRAME_RECEIVED):
                # print(f"State machine did not transition on FRAME_RECEIVED from {self.state_machine.get_state()}")
                pass # Or handle if transition is critical

            # Perform detection
            # Ensure frame is in the correct format if needed (e.g., RGB for some models)
            results = self.model(frame, conf=self.config.conf_threshold, device=self.device) # Pass device
            detections = []
            
            if len(results) > 0:
                result = results[0] # Assuming single image processing
                boxes = result.boxes # Accessing detected boxes
                
                # Process detected objects
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    confidence = box.conf[0].item() # Confidence score
                    class_id = int(box.cls[0].item()) # Class ID
                    
                    # Get category info using WasteClassifier
                    category_id, description = self.waste_classifier.get_category_info(class_id)
                    display_text = f"{category_id}({description})" # Text for display
                    
                    # Calculate area and direction
                    area = self._calculate_area(x1, y1, x2, y2)
                    w, h = x2 - x1, y2 - y1
                    direction = self._determine_direction(w, h)
                    
                    # Create detection object
                    detection = Detection(
                        class_id=class_id, # Use original class_id from model
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=center_x, center_y=center_y,
                        confidence=confidence,
                        display_text=display_text,
                        area=area,
                        direction=direction
                    )
                    
                    # Add to detections list
                    detections.append(detection)
                    
                    # Visualize if debug is enabled
                    if self.config.debug_window:
                        color = self.colors.get(class_id, (255, 255, 255)) # Default to white if color not found
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1) # Green circle for center
                        
                        label = f"{display_text} {confidence:.2f} A:{area} D:{direction}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # White text
                
                # Sort by area (larger objects first)
                detections.sort(key=lambda x: x.area, reverse=True)
                
                # Add to processing queue
                with self.queue_lock:
                    # Clear current queue to process only newest detections from this frame
                    self.processing_queue = [] 
                    
                    # Add new detections
                    for detection_obj in detections: # Renamed to avoid conflict with outer 'detection'
                        # Always process detection (removed time-based filtering from here)
                        self.processing_queue.append(detection_obj)
                        
                        if self.config.debug_window:
                            print(f"Added to queue: {detection_obj.display_text}, Area: {detection_obj.area}, Direction: {detection_obj.direction}")
            
            # Update state machine
            self.state_machine.trigger(DetectionEvent.DETECTION_COMPLETED, detections)
            
            return frame # Return the annotated frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            self.event_bus.publish(DetectionEvent.ERROR_OCCURRED, str(e))
            return frame # Return original frame on error
    
    def _calculate_area(self, x1, y1, x2, y2):
        """Calculate area of detection box"""
        return abs((x2 - x1) * (y2 - y1))
    
    def _determine_direction(self, w, h):
        """Determine direction based on width and height"""
        return 1 if w > h else 2 # 1 for horizontal, 2 for vertical or square
    
    def _should_process_detection(self, detection):
        """
        Determine if a detection should be processed.
        Note: As requested, we'll always return True to process all detections.
        This function can be expanded later for more complex filtering.
        """
        return True
    
    def _process_queue(self):
        """Process detection queue"""
        while self.is_processing:
            detection_to_process = None
            
            with self.queue_lock:
                if self.processing_queue:
                    detection_to_process = self.processing_queue.pop(0) # FIFO
            
            if detection_to_process:
                # Update state machine (transition to SENDING)
                # This event is published to the event bus, which the SerialService listens to.
                # The state machine transition here signifies that the DetectionService is ready to send.
                if self.state_machine.trigger(DetectionEvent.SEND_DETECTION, detection_to_process):
                    # The actual sending is handled by SerialService, triggered by the event below.
                    # This print is for local logging within DetectionService.
                    print(f"\nPreparing to send detection: {detection_to_process.display_text}")
                    print(f"Confidence: {detection_to_process.confidence:.2%}")
                    print(f"Center position: ({detection_to_process.center_x}, {detection_to_process.center_y})")
                    print(f"Target area: {detection_to_process.area} pixels^2")
                    print(f"Direction: {detection_to_process.direction}")
                    print("-" * 30)
                
                # Publish for serial service to handle the actual sending
                # This is the key event that SerialService will pick up.
                self.event_bus.publish(DetectionEvent.SEND_DETECTION, detection_to_process)
                
                # Wait for specified interval (0 by default, meaning no artificial delay here)
                time.sleep(self.config.send_interval)
            else:
                # Queue empty, sleep to avoid high CPU usage
                time.sleep(0.01) # Small sleep when queue is empty
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up DetectionService resources...")
        self.is_processing = False # Signal the processing thread to stop
        
        if hasattr(self, "process_thread") and self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0) # Wait for thread to finish
                print("Detection processing thread terminated.")
            except Exception as e:
                print(f"Error terminating processing thread: {str(e)}")
        # Add any model-specific cleanup if necessary (e.g., del self.model)


# ============================================================
# Statistics Manager
# ============================================================

class StatisticsManager:
    """Manages detection statistics"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.garbage_count = 0
        self.detected_items = [] # List to store details of detected items
        
        # Subscribe to detection events (specifically when a detection is sent)
        # Changed to DETECTION_SENT to count after successful send, or SEND_DETECTION if counting when queued for send.
        # Using SEND_DETECTION as per original logic.
        self.event_bus.subscribe(DetectionEvent.SEND_DETECTION, self.update_statistics)
    
    def update_statistics(self, detection):
        """Update statistics for a detected item"""
        self.garbage_count += 1
        self.detected_items.append({
            "count": self.garbage_count, # Overall count
            "type": detection.display_text, # Type of waste
            "quantity": 1, # Assuming 1 item per detection for now
            "status": "Correct", # Placeholder status, can be updated based on feedback
            "timestamp": detection.timestamp # Store timestamp of detection
        })
        # print(f"Statistics updated: Total items = {self.garbage_count}") # Optional debug print
    
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
        # Load global configuration or use provided config
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
        use_gpu, device_info = setup_gpu() # Assuming setup_gpu returns (bool, str)
        print("\nDevice Information:")
        print(device_info)
        print("-" * 30)
    
    def run(self):
        """Run the application"""
        # Initialize camera
        cap = find_camera(self.config.camera_width, self.config.camera_height)
        if not cap:
            print("Could not find camera. Exiting.")
            return
        
        print("\nSystem Starting:")
        print("- Camera is ready.")
        print(f"- Debug Window: {'Enabled' if self.config.debug_window else 'Disabled'}")
        print(f"- Serial Output: {'Enabled' if self.config.enable_serial else 'Disabled'}")
        if self.config.enable_serial and self.serial_service.port is None:
             print(f"  - Serial Port ({self.config.stm32_port}): FAILED TO INITIALIZE")
        elif self.config.enable_serial:
             print(f"  - Serial Port ({self.config.stm32_port}): Initialized")

        print(f"- Network Debug Port: {'Enabled (Port ' + str(self.config.debug_port) + ')' if self.config.enable_debug_port else 'Disabled'}")
        print("- Press 'q' to exit the program.")
        print("-" * 30)
        
        try:
            while True:
                # Read frame from camera
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame from camera. Exiting loop.")
                    break
                
                # Crop frame if crop_points are defined
                if self.config.crop_points and isinstance(self.config.crop_points, list) and len(self.config.crop_points) > 0 : # Check if it's a valid list of points
                    frame = crop_frame(frame, points=self.config.crop_points)
                
                # Process frame for detection
                processed_frame = self.detection_service.detect(frame) # Renamed to avoid confusion
                
                # Display frame
                if self.config.debug_window:
                    window_name = "YOLO Waste Detection"
                    cv2.imshow(window_name, processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n'q' pressed. Exiting program normally.")
                        break
                        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting program.")
            
        finally:
            # Clean up resources
            self.cleanup(cap)
    
    def cleanup(self, cap):
        """Clean up all resources"""
        print("\nCleaning up application resources...")
        # Clean up components
        if hasattr(self, 'detection_service'):
            self.detection_service.cleanup()
        if hasattr(self, 'serial_service'):
            self.serial_service.cleanup()
        
        # Release camera
        if cap:
            cap.release()
            print("Camera released.")
        
        # Close OpenCV windows
        if self.config.debug_window:
            cv2.destroyAllWindows()
            print("OpenCV windows closed.")
        print("Cleanup complete.")


# ============================================================
# Main Function
# ============================================================

def main():
    """Main function to run the application"""
    # Get the directory of the current script
    script_dir = get_script_directory()
    # Construct the full path to the model file
    full_model_path = os.path.join(script_dir, model_path)

    # Create application with default config, potentially overridden by global vars
    app_config = Config(
        debug_window=DEBUG_WINDOW,
        enable_serial=ENABLE_SERIAL,
        conf_threshold=CONF_THRESHOLD,
        model_path=full_model_path, # Use the constructed full path
        stm32_port=STM32_PORT,
        stm32_baud=STM32_BAUD,
        enable_debug_port=ENABLE_DEBUG_PORT,
        debug_port=DEBUG_PORT
        # crop_points can be set here if needed, e.g.
        # crop_points=[(465, 0), (1146, 720)] if CROP_POINTS_ENABLED else False
    )
    
    app = WasteDetectionApp(config=app_config)
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
