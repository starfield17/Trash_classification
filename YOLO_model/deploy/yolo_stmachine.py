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
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# ============================================================
# Global Variables(默认变量)
# ============================================================
# Default configuration values
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9
model_path = "yolov12n_e300.pt"
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200

# ============================================================
# Configuration(配置参数)
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
    
    stm32_port: str = "/dev/ttyUSB0"#choose any serial you want 
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
# Event System
# ============================================================

class DetectionState(Enum):
    """States for the detection state machine"""
    IDLE = auto()
    DETECTING = auto()
    PROCESSING = auto()
    SENDING = auto()
    ERROR = auto()

class DetectionEvent(Enum):
    """Events for the detection state machine"""
    FRAME_RECEIVED = auto()
    DETECTION_COMPLETED = auto()
    SEND_DETECTION = auto()
    DETECTION_SENT = auto()
    ERROR_OCCURRED = auto()
    RESET = auto()


class EventBus:
    """A simple event bus for decoupling components"""
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, *args, **kwargs):
        """Publish an event to all subscribers"""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)


# ============================================================
# State Machine
# ============================================================

class StateMachine:
    """A generic state machine implementation"""
    
    def __init__(self, initial_state):
        self.state = initial_state
        self.transitions = {}
        self.callbacks = {}
        
    def add_transition(self, from_state, event, to_state, callback=None):
        """Add a transition to the state machine"""
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][event] = to_state
        
        if callback:
            if (from_state, event) not in self.callbacks:
                self.callbacks[(from_state, event)] = []
            self.callbacks[(from_state, event)].append(callback)
    
    def trigger(self, event, *args, **kwargs):
        """Trigger an event on the state machine"""
        if self.state in self.transitions and event in self.transitions[self.state]:
            # Execute callbacks
            if (self.state, event) in self.callbacks:
                for callback in self.callbacks[(self.state, event)]:
                    callback(*args, **kwargs)
            
            # Transition to the new state
            old_state = self.state
            self.state = self.transitions[self.state][event]
            return True
        return False
    
    def get_state(self):
        """Get the current state of the state machine"""
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
        self.zero_mapping = max(waste_classifier.class_names.keys()) + 1
        print(f"类别0将被映射到: {self.zero_mapping}")
        print(f"整体偏移量: +{self.config.class_id_offset}")
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
            print(f"STM32串口已初始化: {self.config.stm32_port}")
            
            # Start queue processing thread
            self.queue_thread = threading.Thread(target=self._process_queue)
            self.queue_thread.daemon = True
            self.queue_thread.start()
            print("串口发送队列处理线程已启动")
        except Exception as e:
            print(f"STM32串口初始化失败: {str(e)}")
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
        
        # Split coordinates into high and low bytes
        x_low = detection.center_x & 0xFF
        x_high = (detection.center_x >> 8) & 0xFF
        y_low = detection.center_y & 0xFF
        y_high = (detection.center_y >> 8) & 0xFF
        
        with self.queue_lock:
            # Limit queue size to avoid memory issues
            if len(self.send_queue) >= 10:
                self.send_queue = self.send_queue[-9:]
                print("警告: 发送队列已满，丢弃旧数据")
            
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
                print(f"队列处理异常: {str(e)}")
            
            # Short sleep to avoid high CPU usage
            time.sleep(0.01)
    
    def _send_next_item(self):
        """Send the next item in the queue"""
        # Check serial port status
        if not self.port:
            return
        
        if not self.port.is_open:
            try:
                print("尝试重新打开串口...")
                self.port.open()
                print("串口重新打开成功")
            except Exception as e:
                print(f"串口重新打开失败: {str(e)}")
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
                print("\n----- 串口发送详细数据 [DEBUG] -----")
                print(f"十六进制数据: {' '.join([f'0x{b:02X}' for b in data])}")
                
                print("原始数据包结构:")
                print(f"  [0] 0x{data[0]:02X} - 帧头1")
                print(f"  [1] 0x{data[1]:02X} - 帧头2")
                print(f"  [2] 0x{data[2]:02X} - 类别ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})")
                print(f"  [3] 0x{data[3]:02X} - X坐标高8位")
                print(f"  [4] 0x{data[4]:02X} - X坐标低8位")
                print(f"  [5] 0x{data[5]:02X} - Y坐标高8位")
                print(f"  [6] 0x{data[6]:02X} - Y坐标低8位")
                print(f"  [7] 0x{data[7]:02X} - 方向 (1: w>h, 2: w<h)")
                print(f"  [8] 0x{data[8]:02X} - 帧尾")
                print(f"数据包总长度: {len(data)} 字节，实际写入: {bytes_written} 字节")
                print(f"原始分类ID: {data_to_send['orig_class']} (十进制) -> {data_to_send['class_id']} (发送值)")
                print(f"原始X坐标: {data_to_send['orig_x']} -> 拆分: 低8位=0x{data_to_send['x_low']:02X}, 高8位=0x{data_to_send['x_high']:02X}")
                print(f"原始Y坐标: {data_to_send['orig_y']} -> 拆分: 低8位=0x{data_to_send['y_low']:02X}, 高8位=0x{data_to_send['y_high']:02X}")
                print(f"方向: {data_to_send['direction']}")
                print(f"数据在队列中等待时间: {time.time() - data_to_send['timestamp']:.3f}秒")
                print("-" * 50)
            
            # Notify that detection was sent
            self.event_bus.publish(DetectionEvent.DETECTION_SENT, data_to_send)
            
        except serial.SerialTimeoutException:
            print("串口写入超时，可能是设备未响应")
            # Put back in queue for retry
            with self.queue_lock:
                self.send_queue.insert(0, data_to_send)
        
        except Exception as e:
            print(f"串口发送错误: {str(e)}")
            # Retry sending data
            with self.queue_lock:
                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # Maximum 3 retries
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send)
                    print(f"数据将重试发送，第{retry_count}次尝试")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        print("正在清理串口资源...")
        
        # Wait for queue thread to end
        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("队列处理线程已终止")
        
        # Close serial port
        if self.port and self.port.is_open:
            self.port.close()
            print("串口已关闭")


# ============================================================
# Detection Service
# ============================================================

class DetectionService:
    """Handles object detection using YOLO"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.state_machine = StateMachine(DetectionState.IDLE)
        self.waste_classifier = WasteClassifier()
        
        # Colors for visualization
        self.colors = {
            0: (86, 180, 233),   # 厨余垃圾 - 蓝色
            1: (230, 159, 0),    # 可回收垃圾 - 橙色
            2: (240, 39, 32),    # 有害垃圾 - 红色
            3: (0, 158, 115),    # 其他垃圾 - 绿色
        }
        
        # Detection state
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.is_processing = False
        self.last_detection_dict = {}
        
        # Configure state machine
        self._setup_state_machine()
        
        # Load YOLO model
        self._load_model()
        
        # Start processing thread
        self.start_processing_thread()
    
    def _setup_state_machine(self):
        """Set up state machine transitions"""
        # Idle to Detecting
        self.state_machine.add_transition(
            DetectionState.IDLE,
            DetectionEvent.FRAME_RECEIVED,
            DetectionState.DETECTING
        )
        
        # Detecting to Processing
        self.state_machine.add_transition(
            DetectionState.DETECTING,
            DetectionEvent.DETECTION_COMPLETED,
            DetectionState.PROCESSING
        )
        
        # Processing to Sending
        self.state_machine.add_transition(
            DetectionState.PROCESSING,
            DetectionEvent.SEND_DETECTION,
            DetectionState.SENDING
        )
        
        # Sending back to Idle
        self.state_machine.add_transition(
            DetectionState.SENDING,
            DetectionEvent.DETECTION_SENT,
            DetectionState.IDLE
        )
        
        # Error handling
        for state in DetectionState:
            if state != DetectionState.ERROR:
                self.state_machine.add_transition(
                    state,
                    DetectionEvent.ERROR_OCCURRED,
                    DetectionState.ERROR
                )
        
        # Reset from error
        self.state_machine.add_transition(
            DetectionState.ERROR,
            DetectionEvent.RESET,
            DetectionState.IDLE
        )
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(self.config.model_path)
            print(f"YOLO模型已加载: {self.config.model_path}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            self.event_bus.publish(DetectionEvent.ERROR_OCCURRED, str(e))
    
    def start_processing_thread(self):
        """Start the detection processing thread"""
        self.is_processing = True
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("检测顺序处理线程已启动")
    
    def detect(self, frame):
        """Detect objects in a frame"""
        try:
            # Update state machine
            self.state_machine.trigger(DetectionEvent.FRAME_RECEIVED)
            
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
                    
                    # Get category info
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
                    
                    # Add to detections list
                    detections.append(detection)
                    
                    # Visualize if debug is enabled
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
                    
                    # Add new detections
                    for detection in detections:
                        # Always process detection (removed time-based filtering)
                        self.processing_queue.append(detection)
                        
                        if self.config.debug_window:
                            print(f"加入队列: {detection.display_text}, 面积: {detection.area}, 方向: {detection.direction}")
            
            # Update state machine
            self.state_machine.trigger(DetectionEvent.DETECTION_COMPLETED, detections)
            
            return frame
            
        except Exception as e:
            print(f"处理帧异常: {str(e)}")
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
                
                print(f"\n发送检测: {detection_to_process.display_text}")
                print(f"置信度: {detection_to_process.confidence:.2%}")
                print(f"中心点位置: ({detection_to_process.center_x}, {detection_to_process.center_y})")
                print(f"目标面积: {detection_to_process.area} 像素^2")
                print(f"方向: {detection_to_process.direction}")
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
        print("正在清理DetectionService资源...")
        self.is_processing = False
        
        if hasattr(self, "process_thread") and self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0)
                print("检测处理线程已终止")
            except Exception as e:
                print(f"终止处理线程出错: {str(e)}")


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
            "status": "正确",
        })
    
    def get_statistics(self):
        """Get current statistics"""
        return {
            "total_count": self.garbage_count,
            "items": self.detected_items
        }


# ============================================================
# Application Class
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
        print("\n设备信息:")
        print(device_info)
        print("-" * 30)
    
    def run(self):
        """Run the application"""
        # Initialize camera
        cap = find_camera()
        if not cap:
            print("找不到摄像头")
            return
        print("\n系统启动:")
        print("- 摄像头已就绪")
        print(f"- 调试窗口: {'开启' if self.config.debug_window else '关闭'}")
        print(f"- 串口输出: {'开启' if self.config.enable_serial else '关闭'}")
        print("- 按 'q' 键退出程序")
        print("-" * 30)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("错误: 无法读取摄像头画面")
                    break
                
                # Crop frame
                if self.config.crop_points:
                    frame = crop_frame(frame, points=self.config.crop_points)
                
                # Process frame
                frame = self.detection_service.detect(frame)
                
                # Display frame
                if self.config.debug_window:
                    window_name = "YOLO_detect"
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n程序正常退出")
                        break
                        
        except KeyboardInterrupt:
            print("\n检测到键盘中断,程序退出")
            
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
# Main Function
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
