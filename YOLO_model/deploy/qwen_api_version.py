import cv2
import torch
import serial
import numpy as np
import threading
import time
import subprocess
import sys
import os
import json
import base64
import re
from openai import OpenAI
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# 全局控制变量

DEBUG_WINDOW = False
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9  # 置信度阈值

# 串口配置
# 可用串口对应关系(raspberrypi)：
# 串口名称  | TX引脚  | RX引脚
# ttyS0    | GPIO14 | GPIO15
# ttyAMA2  | GPIO0  | GPIO1
# ttyAMA3  | GPIO4  | GPIO5
# ttyAMA4  | GPIO8  | GPIO9
# ttyAMA5  | GPIO12 | GPIO13

STM32_PORT = "/dev/ttyUSB0"  # choose serial if you want
STM32_BAUD = 115200
CAMERA_WIDTH = 1280  # 摄像头宽度
CAMERA_HEIGHT = 720  # 摄像头高度
MAX_SERIAL_VALUE = 255  # 串口发送的最大值


class SerialManager:
    def __init__(self):
        self.stm32_port = None
        self.is_running = True
        self.last_stm32_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1  # 最小发送间隔（秒）

        # 当前帧检测结果

        self.current_detections = []

        # 串口发送队列相关

        self.send_queue = []
        self.queue_lock = threading.Lock()
        self.MAX_QUEUE_SIZE = 50  # 队列最大长度

        # 垃圾计数和记录相关

        self.garbage_count = 0
        self.detected_items = []

        # 防重复计数和稳定性检测相关

        self.last_count_time = 0
        self.COUNT_COOLDOWN = 5.0
        self.is_counting_locked = False
        self.last_detected_type = None

        # 稳定性检测相关

        self.current_detection = None
        self.detection_start_time = 0
        self.STABILITY_THRESHOLD = 1.0
        self.stable_detection = False
        self.detection_lost_time = 0
        self.DETECTION_RESET_TIME = 0.5

        # 分类映射

        waste_classifier = WasteClassifier()
        self.zero_mapping = max(waste_classifier.class_names.keys()) + 1
        print(f"类别0将被映射到: {self.zero_mapping}")

        # 初始化STM32串口

        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(
                    STM32_PORT, STM32_BAUD, timeout=0.1, write_timeout=0.1
                )
                print(f"STM32串口已初始化: {STM32_PORT}")

                # 启动队列处理线程

                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("串口发送队列处理线程已启动")
            except Exception as e:
                print(f"STM32串口初始化失败: {str(e)}")
                self.stm32_port = None

    def check_detection_stability(self, garbage_type):
        """检查检测的稳定性"""
        current_time = time.time()

        # 如果检测到了新的物体类型，或者检测中断超过重置时间

        if garbage_type != self.current_detection or (
            current_time - self.detection_lost_time > self.DETECTION_RESET_TIME
            and self.detection_lost_time > 0
        ):
            # 重置检测状态

            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        # 如果已经达到稳定识别时间

        if (
            current_time - self.detection_start_time >= self.STABILITY_THRESHOLD
            and not self.stable_detection
        ):
            self.stable_detection = True
            return True
        return self.stable_detection

    def can_count_new_garbage(self, garbage_type):
        """检查是否可以计数新垃圾"""
        current_time = time.time()

        # 检查稳定性

        if not self.check_detection_stability(garbage_type):
            return False
        # 如果是新的垃圾类型，重置锁定状态

        if garbage_type != self.last_detected_type:
            self.is_counting_locked = False
            self.last_detected_type = garbage_type
        # 检查是否在冷却时间内

        if self.is_counting_locked:
            if current_time - self.last_count_time >= self.COUNT_COOLDOWN:
                self.is_counting_locked = False  # 解除锁定
            else:
                return False
        return True

    def update_garbage_count(self, garbage_type):
        """更新垃圾计数"""
        if not self.can_count_new_garbage(garbage_type):
            return
        self.garbage_count += 1
        self.detected_items.append(
            {
                "count": self.garbage_count,
                "type": garbage_type,
                "quantity": 1,
                "status": "正确",
            }
        )

        # 更新计数相关的状态

        self.last_count_time = time.time()
        self.is_counting_locked = True
        self.last_detected_type = garbage_type

    def clear_detections(self):
        """清空当前帧的检测结果"""
        self.current_detections = []

    def add_detection(self, class_id, center_x, center_y):
        """添加新的检测结果到临时列表"""
        self.current_detections.append((class_id, center_x, center_y))

    def queue_processor_thread(self):
        """队列处理线程，定期从队列中取出数据发送"""
        print("队列处理线程已启动")
        while self.is_running:
            try:
                self._process_queue_batch()
            except Exception as e:
                print(f"队列处理异常: {str(e)}")
            # 控制处理频率

            time.sleep(self.MIN_SEND_INTERVAL / 2)

    def _process_queue_batch(self):
        """处理发送队列中的数据"""
        # 1. 检查串口状态

        if not self.stm32_port:
            print("错误: 串口对象不存在")
            return
        if not self.stm32_port.is_open:
            try:
                print("尝试重新打开串口...")
                self.stm32_port.open()
                print("串口重新打开成功")
            except Exception as e:
                print(f"串口重新打开失败: {str(e)}")
                return
        # 2. 控制发送频率

        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return  # 未到发送间隔，等待下次处理
        # 3. 从队列中获取一条数据

        data_to_send = None
        with self.queue_lock:
            if self.send_queue:
                data_to_send = self.send_queue.pop(0)
        if not data_to_send:
            return
        try:
            # 4. 数据准备
            # 简化：不重置缓冲区，可能会导致额外延迟
            # self.stm32_port.reset_input_buffer()
            # self.stm32_port.reset_output_buffer()
            # time.sleep(0.01)

            # 5. 组装数据包，添加包头包尾标识

            data = bytes(
                [
                    data_to_send["class_id"],
                    data_to_send["x"],
                    data_to_send["y"],
                ]
            )

            # 6. 发送数据

            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            self.last_stm32_send_time = current_time

            if DEBUG_WINDOW:
                print("\n----- 串口发送详细数据 [DEBUG] -----")
                print(f"十六进制数据: {' '.join([f'0x{b:02X}' for b in data])}")

                # 添加更直观的原始数据包展示

                print("原始数据包结构:")
                print(
                    f"  [0] 0x{data[0]:02X} - 类别ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})"
                )
                print(
                    f"  [1] 0x{data[1]:02X} - X坐标 ({data_to_send['orig_x']} -> {data_to_send['x']})"
                )
                print(
                    f"  [2] 0x{data[2]:02X} - Y坐标 ({data_to_send['orig_y']} -> {data_to_send['y']})"
                )
                print(f"数据包总长度: {len(data)} 字节，实际写入: {bytes_written} 字节")
                print(
                    f"原始分类ID: {data_to_send['orig_class']} (十进制) -> {data_to_send['class_id']} (发送值)"
                )
                print(
                    f"原始坐标: ({data_to_send['orig_x']}, {data_to_send['orig_y']}) -> 缩放后: (0x{data_to_send['x']:02X}, 0x{data_to_send['y']:02X})"
                )
                print(
                    f"数据在队列中等待时间: {current_time - data_to_send['timestamp']:.3f}秒"
                )
                print("-" * 50)
        except serial.SerialTimeoutException:
            print("串口写入超时，可能是设备未响应")
            # 超时的数据重新放回队列，保持数据不丢失

            with self.queue_lock:
                self.send_queue.insert(0, data_to_send)
        except Exception as e:
            print(f"串口发送错误: {str(e)}")
            # 发送失败也考虑重新放回队列

            with self.queue_lock:
                # 只在重试次数不超过限制时放回队列

                retry_count = data_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # 最多重试3次
                    data_to_send["retry"] = retry_count
                    self.send_queue.insert(0, data_to_send)
                    print(f"数据将重试发送，第{retry_count}次尝试")
        except serial.SerialTimeoutException:
            print("串口写入超时，将重新入队...")
            # 超时的批次重新放回队列头部

            with self.queue_lock:
                # 增加重试次数

                retry_count = batch_to_send.get("retry", 0) + 1
                if retry_count <= 3:  # 最多重试3次
                    batch_to_send["retry"] = retry_count
                    # 重新放入队列头部

                    self.send_queue.insert(0, batch_to_send)
                    print(f"批次将重试发送，第{retry_count}次尝试")
                else:
                    print(f"批次重试次数已达上限，丢弃该批次")
        except Exception as e:
            print(f"串口发送错误: {str(e)}")
            # 其他错误时也考虑重试

            with self.queue_lock:
                retry_count = batch_to_send.get("retry", 0) + 1
                if retry_count <= 3:
                    batch_to_send["retry"] = retry_count
                    self.send_queue.insert(0, batch_to_send)
                    print(f"错误后重试，第{retry_count}次尝试")

    def send_to_stm32(self, class_id, center_x, center_y):
        """发送数据到STM32，使用队列确保数据可靠传输"""
        # 检查串口和队列线程状态

        if not self.stm32_port or not self.stm32_port.is_open:
            print("警告: 串口未初始化或未打开，无法发送数据")
            return False
        # 检查队列线程是否在运行

        if hasattr(self, "queue_thread") and not self.queue_thread.is_alive():
            print("警告: 队列处理线程未运行，尝试重新启动")
            try:
                self.queue_thread = threading.Thread(target=self.queue_processor_thread)
                self.queue_thread.daemon = True
                self.queue_thread.start()
                print("串口发送队列处理线程已重新启动")
            except Exception as e:
                print(f"启动队列线程失败: {str(e)}")
                return False
        # 确保数据在有效范围内

        if class_id == 0:
            mapped_class_id = self.zero_mapping
        else:
            mapped_class_id = class_id
        # 强制限制所有值在0-255范围内

        mapped_class_id = min(255, max(0, mapped_class_id))
        x_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_x * MAX_SERIAL_VALUE / CAMERA_WIDTH))
        )
        y_scaled = min(
            MAX_SERIAL_VALUE, max(0, int(center_y * MAX_SERIAL_VALUE / CAMERA_HEIGHT))
        )

        # 添加到发送队列

        with self.queue_lock:
            # 限制队列大小，避免内存占用过大

            if len(self.send_queue) >= 10:  # 减小队列大小上限，避免积压太多
                # 保留最新的数据，丢弃旧数据

                self.send_queue = self.send_queue[-9:]
                print("警告: 发送队列已满，丢弃旧数据")
            # 将数据添加到队列，包含原始值和缩放后的值

            self.send_queue.append(
                {
                    "class_id": mapped_class_id,
                    "x": x_scaled,
                    "y": y_scaled,
                    "timestamp": time.time(),
                    "orig_x": center_x,
                    "orig_y": center_y,
                    "orig_class": class_id,
                    "retry": 0,  # 初始重试次数
                }
            )
        return True

    def cleanup(self):
        """清理资源"""
        self.is_running = False
        print("正在清理资源...")

        # 等待队列处理线程结束

        if hasattr(self, "queue_thread") and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("队列处理线程已终止")
        # 关闭串口

        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            print("串口已关闭")


class Qwen25VLDetector:
    def __init__(self):
        # Initialize the OpenAI client for Qwen2.5-VL API
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # Initialize SerialManager for communication with STM32
        self.serial_manager = SerialManager()
        
        # Initialize detection parameters
        self.last_detection_time = 0
        self.detection_interval = 0.5  # 0.5 second throttling
        self.last_detection_dict = {}
        self.min_position_change = 20  # Position change threshold in pixels
        
        # Define class names and colors (matching the original code)
        self.class_names = {
            0: "厨余垃圾",
            1: "可回收垃圾",
            2: "有害垃圾", 
            3: "其他垃圾",
        }
        self.colors = {
            0: (86, 180, 233),  # 厨余垃圾 - 蓝色
            1: (230, 159, 0),   # 可回收垃圾 - 橙色
            2: (240, 39, 32),   # 有害垃圾 - 红色
            3: (0, 158, 115),   # 其他垃圾 - 绿色
        }
        
        # Mapping from waste names to class IDs
        self.name_to_id = {
            "厨余垃圾": 0,
            "可回收垃圾": 1,
            "有害垃圾": 2,
            "其他垃圾": 3,
        }
        
        # Also add some common alternative names 
        self.name_to_id.update({
            "kitchen waste": 0,
            "food waste": 0,
            "organic waste": 0,
            "recyclable waste": 1,
            "recyclables": 1,
            "hazardous waste": 2,
            "harmful waste": 2,
            "dangerous waste": 2,
            "other waste": 3,
            "residual waste": 3,
        })
        
        print("Qwen2.5-VL Detector initialized successfully")
    
    def _should_send_detection(self, class_id, center_x, center_y, confidence):
        """Determine if detection should be sent based on time and position"""
        current_time = time.time()
        
        # Check confidence
        if confidence < CONF_THRESHOLD:
            return False
            
        # Check time interval
        if current_time - self.last_detection_time < self.detection_interval:
            # If time interval is too short, check if there's significant change
            
            # If it's a new class, send it
            if class_id not in self.last_detection_dict:
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True
                
            # Check position change
            last_pos = self.last_detection_dict[class_id]["position"]
            if (abs(center_x - last_pos[0]) > self.min_position_change or
                abs(center_y - last_pos[1]) > self.min_position_change):
                # Position has changed significantly
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True
                
            # Not enough change, don't send
            return False
            
        # Time interval is sufficient, update state and send
        self.last_detection_time = current_time
        self.last_detection_dict[class_id] = {
            "position": (center_x, center_y),
            "time": current_time,
        }
        return True
    
    def _encode_image(self, frame):
        """Convert OpenCV frame to base64 for API"""
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None
        image_bytes = buffer.tobytes()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    
    def _parse_json_from_text(self, text):
        """Extract JSON from text response if needed"""
        # Try to parse the entire text as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Look for JSON-like structure in the text
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
                
        # Try to find JSON object enclosed in curly braces
        json_pattern = r'\{[\s\S]*?\}'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        # If all else fails, return empty dict
        return {}
    
    def _detect_waste(self, frame):
        """Use Qwen API to detect waste in the frame"""
        base64_image = self._encode_image(frame)
        if base64_image is None:
            return []
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a waste detection and classification system. Analyze the image to identify waste objects and classify them into one of these categories: 厨余垃圾, 可回收垃圾, 有害垃圾, 其他垃圾. Be precise with bounding box coordinates."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                },
                            },
                            {"type": "text", "text": "Detect and classify waste objects in this image. For each waste object, provide the following in JSON format:\n1. category: one of [厨余垃圾, 可回收垃圾, 有害垃圾, 其他垃圾]\n2. confidence: a number between 0 and 1\n3. x1, y1, x2, y2: bounding box coordinates\n\nRespond with a JSON object containing a 'detections' array. Example format: {\"detections\": [{\"category\": \"可回收垃圾\", \"confidence\": 0.95, \"x1\": 100, \"y1\": 200, \"x2\": 300, \"y2\": 400}]}"},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1,  # Lower temperature for more consistent responses
            )
            
            # Parse the JSON response
            try:
                content_str = response.choices[0].message.content
                
                # Try to parse JSON from the response text
                result = self._parse_json_from_text(content_str)
                
                # Get detections array from result
                detections = result.get("detections", [])
                
                # If no detections found, check if response is already an array
                if not detections and isinstance(result, list):
                    detections = result
                    
                return detections
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response content: {response.choices[0].message.content}")
                return []
                
        except Exception as e:
            print(f"API call error: {e}")
            return []
    
    def _extract_bounding_box(self, detection):
        """Extract bounding box coordinates from detection object"""
        # Check for standard x1, y1, x2, y2 format
        if all(k in detection for k in ["x1", "y1", "x2", "y2"]):
            return (
                int(float(detection["x1"])),
                int(float(detection["y1"])),
                int(float(detection["x2"])),
                int(float(detection["y2"]))
            )
            
        # Check for bbox or bounding_box array format
        for key in ["bbox", "bounding_box", "box"]:
            if key in detection and isinstance(detection[key], list) and len(detection[key]) >= 4:
                bbox = detection[key]
                return (
                    int(float(bbox[0])),
                    int(float(bbox[1])),
                    int(float(bbox[2])),
                    int(float(bbox[3]))
                )
                
        # Check for x, y, width, height format
        if all(k in detection for k in ["x", "y", "width", "height"]):
            return (
                int(float(detection["x"])),
                int(float(detection["y"])),
                int(float(detection["x"]) + float(detection["width"])),
                int(float(detection["y"]) + float(detection["height"]))
            )
            
        # Default to full frame if no valid bounding box found
        return (0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
    
    def _get_class_id_from_category(self, category):
        """Get class ID from category string"""
        # Try direct mapping
        class_id = self.name_to_id.get(category)
        if class_id is not None:
            return class_id
            
        # Try lowercase
        class_id = self.name_to_id.get(category.lower())
        if class_id is not None:
            return class_id
            
        # Check for partial matches
        for name, id in self.name_to_id.items():
            if name in category or category in name:
                return id
                
        # Default to other waste (3)
        return 3
    
    def detect(self, frame):
        """Process frame and return detected objects"""
        # Get waste detection from API
        detections = self._detect_waste(frame)
        
        if not detections:
            return frame
        
        # Process each detection
        for detection in detections:
            try:
                # Get category, defaulting to "其他垃圾"
                category = detection.get("category", detection.get("class", detection.get("type", "其他垃圾")))
                
                # Get confidence, defaulting to 0.9
                confidence = float(detection.get("confidence", detection.get("score", detection.get("prob", 0.9))))
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = self._extract_bounding_box(detection)
                
                # Ensure coordinates are within frame boundaries
                x1 = max(0, min(x1, CAMERA_WIDTH))
                y1 = max(0, min(y1, CAMERA_HEIGHT))
                x2 = max(0, min(x2, CAMERA_WIDTH))
                y2 = max(0, min(y2, CAMERA_HEIGHT))
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Map category to class ID
                class_id = self._get_class_id_from_category(category)
                
                # Get category information for display
                waste_classifier = WasteClassifier()
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                
                # Get color for visualization
                color = self.colors.get(class_id, (255, 255, 255))
                
                # Visualization (if debug window is enabled)
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{display_text} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Check if detection should be sent
                if self._should_send_detection(class_id, center_x, center_y, confidence):
                    print(f"\n发送检测: {display_text}")
                    print(f"置信度: {confidence:.2%}")
                    print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
                    print(f"中心点位置: ({center_x}, {center_y})")
                    print("-" * 30)
                    
                    # Send detection to STM32
                    self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                    self.serial_manager.update_garbage_count(display_text)
                else:
                    print(f"检测到(不发送): {display_text}, 位置: ({center_x}, {center_y})")
                    
            except Exception as e:
                print(f"Error processing detection: {e}")
                import traceback
                traceback.print_exc()
                
        return frame


def create_qwen_detector():
    """
    Create a Qwen2.5-VL API-based detector
    
    Returns:
        detector: Qwen25VLDetector instance
    """
    try:
        print("Initializing Qwen2.5-VL detector...")
        return Qwen25VLDetector()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Qwen2.5-VL detector: {str(e)}")
      
def main():
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)

    try:
        detector = create_qwen_detector()
    except Exception as e:
        print(f"创建检测器失败: {str(e)}")
        return
    
    cap = find_camera()
    if not cap:
        return
        
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print(f"- 调试窗口: {'开启' if DEBUG_WINDOW else '关闭'}")
    print(f"- 串口输出: {'开启' if ENABLE_SERIAL else '关闭'}")
    print("- 按 'q' 键退出程序")
    print("-" * 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
                
            frame = detector.detect(frame)

            if DEBUG_WINDOW:
                window_name = "QWEN25VL_detect"
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n程序正常退出")
                    break
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        # Clean up resources
        if hasattr(detector, "serial_manager"):
            detector.serial_manager.cleanup()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
