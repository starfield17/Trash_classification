import cv2
import torch
import serial
import numpy as np
import threading
import time
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Import PyTorch libraries required for FasterRCNN
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

# Third party function and class
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# ============================================================
# Global config Variables (high priority)
# ============================================================
# Default configuration values
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.7
model_path = "output/model_final.pth"  # Modify to FastCNN model path
STM32_PORT = "/dev/ttyUSB0" # Choose any serial port you want
STM32_BAUD = 115200
MODEL_TYPE = "resnet50_fpn"  # Model type: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"

# Four-category waste dataset configuration
CLASS_NAMES = ["Kitchen Waste", "Recyclable Waste", "Hazardous Waste", "Other Waste"]

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
    conf_threshold: float = 0.70

    # Model configuration
    model_path: str = "output/model_final.pth"
    model_type: str = "resnet50_fpn"  # New field

    # Serial configuration
    stm32_port: str = "/dev/ttyUSB0"
    stm32_baud: int = 115200

    # Camera configuration
    camera_width: int = 1280
    camera_height: int = 720
    # Can be enable option
    crop_points: bool = False

    # Serial protocol configuration
    serial_header1: int = 0x2C
    serial_header2: int = 0x12
    serial_footer: int = 0x5B

    # Processing configuration
    min_position_change: int = 20
    send_interval: float = 0.0  # No delay between sending detections

# ============================================================
# Event System
# ============================================================

class DetectionState(Enum):
    """
    Enumeration of detection system states
    IDLE: Idle state, waiting for new frame input
    DETECTING: Performing object detection
    PROCESSING: Processing detection results
    SENDING: Sending detection data to downstream devices
    ERROR: System error state
    """
    IDLE = auto()        # Idle state, waiting for new frame
    DETECTING = auto()   # Performing object detection
    PROCESSING = auto()  # Processing detection results
    SENDING = auto()     # Sending detection data
    ERROR = auto()       # Error state

class DetectionEvent(Enum):
    """
    Enumeration of events that trigger state transitions
    FRAME_RECEIVED: Received a new video frame
    DETECTION_COMPLETED: Object detection completed
    SEND_DETECTION: Send detection results
    DETECTION_SENT: Detection results sent
    ERROR_OCCURRED: An error occurred
    RESET: Reset the system
    """
    FRAME_RECEIVED = auto()     # Received new video frame
    DETECTION_COMPLETED = auto() # Object detection completed
    SEND_DETECTION = auto()     # Event to send detection results
    DETECTION_SENT = auto()     # Detection results sent
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
        # Stores mapping from event type to subscriber callback functions
        self.subscribers = {}

    def subscribe(self, event_type, callback):
        """
        Subscribe to a specific type of event
        @param event_type: Event type, usually a DetectionEvent enum value
        @param callback: Callback function to be called when the event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type, *args, **kwargs):
        """
        Publish an event to all subscribers
        @param event_type: Type of event to publish
        @param args, kwargs: Arguments to pass to subscriber callback functions
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
    Generic state machine implementation
    - Manages system state transitions
    - Events trigger state transitions
    - Executes callback functions during state transitions
    """

    def __init__(self, initial_state):
        """
        Initialize the state machine
        @param initial_state: Initial state, usually a DetectionState enum value
        """
        self.state = initial_state           # Current state
        self.transitions = {}                # Stores state transition rules
        self.callbacks = {}                  # Stores callback functions for transitions

    def add_transition(self, from_state, event, to_state, callback=None):
        """
        Add a state transition rule
        @param from_state: Starting state
        @param event: Event that triggers the transition
        @param to_state: Target state
        @param callback: Optional callback function, executed during state transition
        """
        # If the starting state is not in the transition table, add it
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        # Set the transition rule from the starting state via the event to the target state
        self.transitions[from_state][event] = to_state

        # If a callback function is provided, store it
        if callback:
            # Store callback function for the specific (state, event) pair
            if (from_state, event) not in self.callbacks:
                self.callbacks[(from_state, event)] = []
            self.callbacks[(from_state, event)].append(callback)

    def trigger(self, event, *args, **kwargs):
        """
        Trigger an event, attempting to execute a state transition
        @param event: Event to trigger
        @param args, kwargs: Arguments to pass to callback functions
        @return: True if state transition is successful, False otherwise
        """
        # Check if there is a transition rule for the current state and event
        if self.state in self.transitions and event in self.transitions[self.state]:
            # Execute all callbacks associated with this state and event
            if (self.state, event) in self.callbacks:
                for callback in self.callbacks[(self.state, event)]:
                    callback(*args, **kwargs)

            # Execute state transition
            old_state = self.state
            self.state = self.transitions[self.state][event]
            return True
        return False

    def get_state(self):
        """
        Get the current state
        @return: Current state value
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
    direction: int
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
        self.zero_mapping = 4  # max(waste_classifier.class_names.keys()) + 1
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
                print("Warning: Send queue is full, discarding old data")
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
                print(f"Serial port reopen failed: {str(e)}")
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
                print(f"Hexadecimal data: {' '.join([f'0x{b:02X}' for b in data])}")

                print("Original data packet structure:")
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
                print(f"Original X Coordinate: {data_to_send['orig_x']} -> Split: Low 8 bits=0x{data_to_send['x_low']:02X}, High 8 bits=0x{data_to_send['x_high']:02X}")
                print(f"Original Y Coordinate: {data_to_send['orig_y']} -> Split: Low 8 bits=0x{data_to_send['y_low']:02X}, High 8 bits=0x{data_to_send['y_high']:02X}")
                print(f"Direction: {data_to_send['direction']}")
                print(f"Data waiting time in queue: {time.time() - data_to_send['timestamp']:.3f} seconds")
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
                    print(f"Data will be resent, attempt {retry_count}")

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
# Detection Service
# ============================================================

class DetectionService:
    """Handles object detection using Faster R-CNN"""

    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.state_machine = StateMachine(DetectionState.IDLE)
        self.waste_classifier = WasteClassifier()

        # Colors for visualization
        self.colors = {
            0: (86, 180, 233),   # Kitchen Waste - Blue
            1: (230, 159, 0),    # Recyclable Waste - Orange
            2: (240, 39, 32),    # Hazardous Waste - Red
            3: (0, 158, 115),    # Other Waste - Green
        }

        # Detection state
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.is_processing = False
        self.last_detection_dict = {}

        # Define image preprocessing transforms
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

        # Configure state machine
        self._setup_state_machine()

        # Load Faster R-CNN model
        self._load_model()

        # Start processing thread
        self.start_processing_thread()

    def _setup_state_machine(self):
        """
        Set up state machine transition rules
        1. Start from IDLE state, transition to DETECTING state when a new video frame is received
        2. After detection is complete in DETECTING state, transition to PROCESSING state
        3. From PROCESSING state, when preparing to send data, transition to SENDING state
        4. After data is sent, transition from SENDING state back to IDLE state, ready for next frame processing
        5. In any state, if an error occurs, transition to ERROR state
        6. From ERROR state, can reset back to IDLE state via RESET event
        """
        # IDLE state -> DETECTING state: When a new video frame is received
        # Starting point of the state machine, indicates the system starts receiving new frames for detection from an idle state
        self.state_machine.add_transition(
            DetectionState.IDLE,                # Starting state: Idle
            DetectionEvent.FRAME_RECEIVED,      # Triggering event: New frame received
            DetectionState.DETECTING            # Target state: Detecting
        )

        # DETECTING state -> PROCESSING state: When detection is complete
        # When object detection is complete, the system needs to process the detection results
        self.state_machine.add_transition(
            DetectionState.DETECTING,           # Starting state: Detecting
            DetectionEvent.DETECTION_COMPLETED, # Triggering event: Detection complete
            DetectionState.PROCESSING           # Target state: Processing results
        )

        # PROCESSING state -> SENDING state: When detection results need to be sent
        # After processing detection results, the system prepares to send data to downstream devices (e.g., STM32)
        self.state_machine.add_transition(
            DetectionState.PROCESSING,          # Starting state: Processing results
            DetectionEvent.SEND_DETECTION,      # Triggering event: Send detection results
            DetectionState.SENDING              # Target state: Sending
        )

        # SENDING state -> IDLE state: When detection results have been sent
        # After sending is complete, the system returns to idle state, waiting for the next frame
        self.state_machine.add_transition(
            DetectionState.SENDING,             # Starting state: Sending
            DetectionEvent.DETECTION_SENT,      # Triggering event: Data sent
            DetectionState.IDLE                 # Target state: Return to idle state
        )

        # Error handling: Any state transitions to ERROR state if an error occurs
        # This is a global error handling mechanism, ensuring the system enters a controllable error state in any exceptional situation
        for state in DetectionState:
            if state != DetectionState.ERROR:   # For all states except ERROR
                self.state_machine.add_transition(
                    state,                       # Any starting state
                    DetectionEvent.ERROR_OCCURRED, # Triggering event: Error occurred
                    DetectionState.ERROR         # Target state: Error state
                )

        # Reset from ERROR state: Return to IDLE state via RESET event
        # Provides a mechanism to recover from an error state and restart the detection process
        self.state_machine.add_transition(
            DetectionState.ERROR,               # Starting state: Error state
            DetectionEvent.RESET,               # Triggering event: Reset
            DetectionState.IDLE                 # Target state: Return to idle state
        )

    def _get_faster_rcnn_model(self, num_classes, model_type):
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
                weights='ResNet18_Weights.IMAGENET1K_V1', # Use new weights argument
                trainable_layers=3
            )

            # Set RPN anchor generator - modify here to match multiple feature map layers
            anchor_generator = AnchorGenerator(
                # Specify separate anchor sizes for each feature map layer
                sizes=((32,), (64,), (128,), (256,), (512,)),
                # Repeat the same aspect ratio configuration for each feature map layer
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )

            # Set RoI pooling size
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

    def _load_model(self):
        """Load the Faster R-CNN model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get number of classes
            num_classes = len(CLASS_NAMES)

            # Build network based on model type
            self.model = self._get_faster_rcnn_model(num_classes, self.config.model_type)

            # Load pre-trained weights
            model_path_abs = os.path.join(get_script_directory(), self.config.model_path) # Ensure absolute path
            state_dict = torch.load(model_path_abs, map_location=self.device)

            # Handle possible key name differences
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"Attempting to load model directly failed, possibly training checkpoint format: {e}")
                # Try loading checkpoint format (if it's a checkpoint with model_state_dict)
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    print("Successfully loaded weights from checkpoint")
                elif 'model' in state_dict: # Common for Detectron2 or MMDetection
                    self.model.load_state_dict(state_dict['model'])
                    print("Successfully loaded weights from 'model' key in checkpoint")
                else:
                    raise RuntimeError("Unable to load model weights, please check model file format")

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            print(f"FasterRCNN model loaded: {self.config.model_path}")
            print(f"Model type: {self.config.model_type}")
            print(f"Number of classes: {num_classes}")
            print(f"Using device: {self.device}")

        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
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
            # Update state machine
            self.state_machine.trigger(DetectionEvent.FRAME_RECEIVED)

            # Convert image format BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess image
            image_tensor = self.transforms(rgb_frame).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                predictions = self.model(image_tensor)

            detections = []

            if len(predictions) > 0:
                prediction = predictions[0]

                # Get predicted bounding boxes, scores, and labels
                boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()

                # Iterate through detection results
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    # Check confidence
                    if score < self.config.conf_threshold:
                        continue

                    # Remember label starts from 1 (0 is background), so actual class is label-1
                    class_id = int(label) - 1

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Get class information
                    if 0 <= class_id < len(CLASS_NAMES):
                        display_text = f"{class_id}({CLASS_NAMES[class_id]})"
                    else:
                        display_text = f"{class_id}(Unknown)"

                    # Calculate area and direction
                    area = self._calculate_area(x1, y1, x2, y2)
                    w, h = x2 - x1, y2 - y1
                    direction = self._determine_direction(w, h)

                    # Create detection object
                    detection = Detection(
                        class_id=class_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=center_x, center_y=center_y,
                        confidence=score,
                        display_text=display_text,
                        area=area,
                        direction=direction
                    )

                    # Add to detection list
                    detections.append(detection)

                    # If debug window is enabled, draw detection results on the image
                    if self.config.debug_window:
                        color = self.colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                        label_text = f"{display_text} {score:.2f} A:{area} D:{direction}" # Renamed variable to avoid conflict
                        (tw, th), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                        )
                        cv2.rectangle(
                            frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1
                        )
                        cv2.putText(
                            frame,
                            label_text,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                        )

                # Sort by area (larger area first)
                detections.sort(key=lambda x: x.area, reverse=True)

                # Add to processing queue
                with self.queue_lock:
                    # Clear current queue
                    self.processing_queue = []

                    # Add new detection results
                    for detection_item in detections: # Renamed variable to avoid conflict
                        self.processing_queue.append(detection_item)

                        if self.config.debug_window:
                            print(f"Added to queue: {detection_item.display_text}, Area: {detection_item.area}, Direction: {detection_item.direction}")

            # Update state machine
            self.state_machine.trigger(DetectionEvent.DETECTION_COMPLETED, detections)

            return frame

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            import traceback
            traceback.print_exc()
            self.event_bus.publish(DetectionEvent.ERROR_OCCURRED, str(e))
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
            detection_to_process = None

            with self.queue_lock:
                if self.processing_queue:
                    detection_to_process = self.processing_queue.pop(0)

            if detection_to_process:
                # Update state machine
                self.state_machine.trigger(DetectionEvent.SEND_DETECTION, detection_to_process)

                print(f"\nSending detection: {detection_to_process.display_text}")
                print(f"Confidence: {detection_to_process.confidence:.2%}")
                print(f"Center point position: ({detection_to_process.center_x}, {detection_to_process.center_y})")
                print(f"Target area: {detection_to_process.area} pixels^2")
                print(f"Direction: {detection_to_process.direction}")
                print("-" * 30)

                # Publish for serial service to handle
                self.event_bus.publish(DetectionEvent.SEND_DETECTION, detection_to_process)

                # Wait for specified interval (0 by default)
                time.sleep(self.config.send_interval)
            else:
                # Queue empty, sleep to avoid high CPU usage
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
# Statistics Manager
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
            "status": "Correct", # "正确" translated to "Correct"
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
        cap = find_camera(self.config.camera_width, self.config.camera_height)
        if not cap:
            print("Camera not found")
            return
        print("\nSystem Startup:")
        print("- Camera ready")
        print(f"- Debug window: {'Enabled' if self.config.debug_window else 'Disabled'}")
        print(f"- Serial output: {'Enabled' if self.config.enable_serial else 'Disabled'}")
        print("- Press 'q' to exit the program")
        print("-" * 30)

        try:
            while True:
                # Read
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read camera frame")
                    break

                # Crop
                if self.config.crop_points:
                    frame = crop_frame(frame, points=self.config.crop_points)

                # Process
                frame = self.detection_service.detect(frame)

                # Display
                if self.config.debug_window:
                    window_name = "FasterRCNN_detect"
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
        if cap: # Check if cap was successfully initialized
            cap.release()

        # Close windows
        if self.config.debug_window:
            cv2.destroyAllWindows()


# ============================================================
# Main Function
# ============================================================

def main():
    """Main function"""
    # Create application with default config
    app = WasteDetectionApp(Config(
        debug_window=DEBUG_WINDOW,
        enable_serial=ENABLE_SERIAL,
        conf_threshold=CONF_THRESHOLD,
        model_path=os.path.join(get_script_directory(), model_path), # Use absolute path
        model_type=MODEL_TYPE,  # Pass model type
        stm32_port=STM32_PORT,
        stm32_baud=STM32_BAUD
    ))

    # Run the application
    app.run()


if __name__ == "__main__":
    main()
