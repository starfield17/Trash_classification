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

# 导入FasterRCNN所需的PyTorch库
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision import transforms as T
from torchvision.transforms import functional as F

# Third party function and class 
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# ============================================================
# Global config Variables(high pri) / 全局配置变量(高优先级)
# ============================================================
# Default configuration values
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.7
model_path = "output/model_final.pth"  # 修改为FastCNN模型路径
STM32_PORT = "/dev/ttyUSB0" #choose any serial you want 
STM32_BAUD = 115200
MODEL_TYPE = "resnet50_fpn"  # 模型类型: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3"

# 四分类垃圾数据集配置
CLASS_NAMES = ["厨余垃圾", "可回收垃圾", "有害垃圾", "其他垃圾"]

# ============================================================
# Default configuration(low pri) / 默认配置参数(低优先级)
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
    model_type: str = "resnet50_fpn"  # 新增字段
    
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
# Event System / 系统事件
# ============================================================

class DetectionState(Enum):
    """
    检测系统的状态枚举
    IDLE: 空闲状态，等待新的帧输入
    DETECTING: 正在进行目标检测
    PROCESSING: 正在处理检测结果
    SENDING: 正在发送检测数据到下游设备
    ERROR: 系统出错状态
    """
    IDLE = auto()        # 空闲状态，等待新的帧
    DETECTING = auto()   # 正在进行对象检测
    PROCESSING = auto()  # 正在处理检测结果
    SENDING = auto()     # 正在发送检测数据
    ERROR = auto()       # 错误状态

class DetectionEvent(Enum):
    """
    触发状态转换的事件枚举
    FRAME_RECEIVED: 接收到新的视频帧
    DETECTION_COMPLETED: 完成目标检测
    SEND_DETECTION: 发送检测结果
    DETECTION_SENT: 检测结果已发送
    ERROR_OCCURRED: 发生错误
    RESET: 重置系统
    """
    FRAME_RECEIVED = auto()     # 收到新的视频帧
    DETECTION_COMPLETED = auto() # 完成了目标检测
    SEND_DETECTION = auto()     # 发送检测结果的事件
    DETECTION_SENT = auto()     # 检测结果已发送
    ERROR_OCCURRED = auto()     # 发生错误
    RESET = auto()              # 重置系统

class EventBus:
    """
    事件总线类：实现了发布-订阅模式，用于系统各组件间的解耦通信
    - 允许不同组件订阅特定类型的事件
    - 当事件发生时，通知所有订阅了该事件的组件
    """
    def __init__(self):
        # 存储事件类型到订阅者回调函数的映射
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """
        订阅特定类型的事件
        @param event_type: 事件类型，通常是DetectionEvent枚举值
        @param callback: 当事件发生时要调用的回调函数
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, *args, **kwargs):
        """
        发布事件到所有订阅者
        @param event_type: 要发布的事件类型
        @param args, kwargs: 传递给订阅者回调函数的参数
        """
        if event_type in self.subscribers:
            # 调用所有订阅了该事件的回调函数
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)


# ============================================================
# State Machine / 状态机
# ============================================================

class StateMachine:
    """
    通用状态机实现
    - 管理系统的状态转换
    - 事件触发状态转换
    - 状态转换时执行回调函数
    """
    
    def __init__(self, initial_state):
        """
        初始化状态机
        @param initial_state: 初始状态，通常是DetectionState枚举值
        """
        self.state = initial_state           # 当前状态
        self.transitions = {}                # 存储状态转换规则
        self.callbacks = {}                  # 存储转换时的回调函数
        
    def add_transition(self, from_state, event, to_state, callback=None):
        """
        添加状态转换规则
        @param from_state: 起始状态
        @param event: 触发转换的事件
        @param to_state: 目标状态
        @param callback: 可选的回调函数，在状态转换时执行
        """
        # 如果起始状态不在转换表中，添加它
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        # 设置从起始状态经过事件到目标状态的转换规则
        self.transitions[from_state][event] = to_state
        
        # 如果提供了回调函数，存储它
        if callback:
            # 为特定的(状态,事件)对存储回调函数
            if (from_state, event) not in self.callbacks:
                self.callbacks[(from_state, event)] = []
            self.callbacks[(from_state, event)].append(callback)
    
    def trigger(self, event, *args, **kwargs):
        """
        触发事件，尝试执行状态转换
        @param event: 要触发的事件
        @param args, kwargs: 传递给回调函数的参数
        @return: 如果状态转换成功返回True，否则返回False
        """
        # 检查当前状态是否有对应事件的转换规则
        if self.state in self.transitions and event in self.transitions[self.state]:
            # 执行与此状态和事件相关的所有回调
            if (self.state, event) in self.callbacks:
                for callback in self.callbacks[(self.state, event)]:
                    callback(*args, **kwargs)
            
            # 执行状态转换
            old_state = self.state
            self.state = self.transitions[self.state][event]
            return True
        return False
    
    def get_state(self):
        """
        获取当前状态
        @return: 当前状态值
        """
        return self.state


# ============================================================
# Data Models / 模型
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
# Serial Communication Service / 串口服务
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
            # Short sleep
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
# Detection Service / 检测服务
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
        
        # 定义图像预处理转换
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
        设置状态机的转换规则
        1. 从IDLE状态开始，当接收到新的视频帧时转到DETECTING状态
        2. 在DETECTING状态完成检测后，转到PROCESSING状态
        3. 从PROCESSING状态准备发送数据时，转到SENDING状态
        4. 数据发送完成后，从SENDING状态回到IDLE状态，准备下一帧处理
        5. 任何状态下如果发生错误，都会转到ERROR状态
        6. 从ERROR状态可以通过RESET事件重置回IDLE状态
        """
        # IDLE状态 -> DETECTING状态：当接收到新的视频帧
        # 状态机的起点，表示系统从空闲状态开始接收新帧进行检测
        self.state_machine.add_transition(
            DetectionState.IDLE,                # 起始状态：空闲
            DetectionEvent.FRAME_RECEIVED,      # 触发事件：收到新帧
            DetectionState.DETECTING            # 目标状态：正在检测
        )
        
        # DETECTING状态 -> PROCESSING状态：当检测完成
        # 当目标检测完成后，系统需要处理检测结果
        self.state_machine.add_transition(
            DetectionState.DETECTING,           # 起始状态：正在检测
            DetectionEvent.DETECTION_COMPLETED, # 触发事件：检测完成
            DetectionState.PROCESSING           # 目标状态：正在处理结果
        )
        
        # PROCESSING状态 -> SENDING状态：当需要发送检测结果
        # 处理完检测结果后，系统准备发送数据到下游设备（如STM32）
        self.state_machine.add_transition(
            DetectionState.PROCESSING,          # 起始状态：正在处理结果
            DetectionEvent.SEND_DETECTION,      # 触发事件：发送检测结果
            DetectionState.SENDING              # 目标状态：正在发送
        )
        
        # SENDING状态 -> IDLE状态：当检测结果已发送
        # 发送完成后，系统回到空闲状态，等待下一帧
        self.state_machine.add_transition(
            DetectionState.SENDING,             # 起始状态：正在发送
            DetectionEvent.DETECTION_SENT,      # 触发事件：数据已发送
            DetectionState.IDLE                 # 目标状态：回到空闲状态
        )
        
        # 错误处理：任何状态下发生错误都会转到ERROR状态
        # 这是一个全局错误处理机制，确保系统在任何异常情况下都能进入可控的错误状态
        for state in DetectionState:
            if state != DetectionState.ERROR:   # 对除ERROR外的所有状态
                self.state_machine.add_transition(
                    state,                       # 任何起始状态
                    DetectionEvent.ERROR_OCCURRED, # 触发事件：发生错误
                    DetectionState.ERROR         # 目标状态：错误状态
                )
        
        # 从ERROR状态重置：通过RESET事件回到IDLE状态
        # 提供从错误状态恢复的机制，重新开始检测流程
        self.state_machine.add_transition(
            DetectionState.ERROR,               # 起始状态：错误状态
            DetectionEvent.RESET,               # 触发事件：重置
            DetectionState.IDLE                 # 目标状态：回到空闲状态
        )

    def _get_faster_rcnn_model(self, num_classes, model_type):
        """获取不同类型的Faster R-CNN模型"""
        # num_classes需要+1，因为0是背景类
        num_classes_with_bg = num_classes + 1
        
        if model_type == "resnet50_fpn":
            # 标准版：ResNet50+FPN
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
        
        elif model_type == "resnet18_fpn":
            # 轻量版：ResNet18+FPN
            backbone = resnet_fpn_backbone(
                'resnet18', 
                pretrained=True, 
                trainable_layers=3
            )
            
            # 设置RPN的锚点生成器
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            
            # 设置RoI池化的大小
            roi_pooler = torch.ops.torchvision.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
            
            # 创建Faster R-CNN模型
            model = FasterRCNN(
                backbone=backbone,
                num_classes=num_classes_with_bg,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        
        elif model_type == "mobilenet_v3":
            # 超轻量版：MobileNetV3
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def _load_model(self):
        """Load the Faster R-CNN model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 获取类别数量
            num_classes = len(CLASS_NAMES)
            
            # 基于模型类型构建网络
            self.model = self._get_faster_rcnn_model(num_classes, self.config.model_type)
            
            # 加载预训练权重
            model_path = os.path.join(get_script_directory(), self.config.model_path)
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理可能的键名差异
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"尝试直接加载模型失败，可能是训练检查点格式: {e}")
                # 尝试加载检查点格式 (如果是带有 model_state_dict 的检查点)
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    print("成功从检查点加载权重")
                else:
                    raise RuntimeError("无法加载模型权重，请检查模型文件格式")
            
            # 将模型移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            print(f"FasterRCNN模型已加载: {self.config.model_path}")
            print(f"模型类型: {self.config.model_type}")
            print(f"类别数量: {num_classes}")
            print(f"使用设备: {self.device}")
            
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
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
            
            # 转换图像格式 BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            image_tensor = self.transforms(rgb_frame).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(image_tensor)
                
            detections = []
            
            if len(predictions) > 0:
                prediction = predictions[0]
                
                # 获取预测的边界框、分数和标签
                boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                
                # 遍历检测结果
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    # 检查置信度
                    if score < self.config.conf_threshold:
                        continue
                    
                    # 记住 label 从 1 开始 (0是背景)，所以实际类别是 label-1
                    class_id = int(label) - 1
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 计算中心点
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # 获取类别信息
                    if 0 <= class_id < len(CLASS_NAMES):
                        display_text = f"{class_id}({CLASS_NAMES[class_id]})"
                    else:
                        display_text = f"{class_id}(未知)"
                    
                    # 计算面积和方向
                    area = self._calculate_area(x1, y1, x2, y2)
                    w, h = x2 - x1, y2 - y1
                    direction = self._determine_direction(w, h)
                    
                    # 创建检测对象
                    detection = Detection(
                        class_id=class_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=center_x, center_y=center_y,
                        confidence=score,
                        display_text=display_text,
                        area=area,
                        direction=direction
                    )
                    
                    # 添加到检测列表
                    detections.append(detection)
                    
                    # 如果启用调试窗口，在图像上绘制检测结果
                    if self.config.debug_window:
                        color = self.colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                        
                        label = f"{display_text} {score:.2f} A:{area} D:{direction}"
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
                
                # 按面积排序（面积大的优先）
                detections.sort(key=lambda x: x.area, reverse=True)
                
                # 添加到处理队列
                with self.queue_lock:
                    # 清空当前队列
                    self.processing_queue = []
                    
                    # 添加新的检测结果
                    for detection in detections:
                        self.processing_queue.append(detection)
                        
                        if self.config.debug_window:
                            print(f"加入队列: {detection.display_text}, 面积: {detection.area}, 方向: {detection.direction}")
            
            # 更新状态机
            self.state_machine.trigger(DetectionEvent.DETECTION_COMPLETED, detections)
            
            return frame
            
        except Exception as e:
            print(f"处理帧异常: {str(e)}")
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
# Statistics Manager / 计数
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
# Application Class / 集成层
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
        cap = find_camera(self.config.camera_width, self.config.camera_height)
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
                # Read
                ret, frame = cap.read()
                if not ret:
                    print("错误: 无法读取摄像头画面")
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
# Main Function / 主函数
# ============================================================

def main():
    """Main function"""
    # Create application with default config
    app = WasteDetectionApp(Config(
        debug_window=DEBUG_WINDOW,
        enable_serial=ENABLE_SERIAL,
        conf_threshold=CONF_THRESHOLD,
        model_path=os.path.join(get_script_directory(), model_path),
        model_type=MODEL_TYPE,  # 传入模型类型
        stm32_port=STM32_PORT,
        stm32_baud=STM32_BAUD
    ))
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
