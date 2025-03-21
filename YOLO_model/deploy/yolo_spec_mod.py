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
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# 全局控制变量

DEBUG_WINDOW = False
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9  # 置信度阈值
model_path = "yolov12n_e300.pt"

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
                
        # 0类别映射处理
        if class_id == 0:
            mapped_class_id = self.zero_mapping
        else:
            mapped_class_id = class_id
            
        # 限制class_id在0-255范围内
        mapped_class_id = min(255, max(0, mapped_class_id))
        
        # 将原坐标拆分为高8位和低8位
        x_low = center_x & 0xFF  # 低8位
        x_high = (center_x >> 8) & 0xFF  # 高8位
        y_low = center_y & 0xFF  # 低8位
        y_high = (center_y >> 8) & 0xFF  # 高8位
    
        # 添加到发送队列
        with self.queue_lock:
            # 限制队列大小，避免内存占用过大
            if len(self.send_queue) >= 10:  # 减小队列大小上限，避免积压太多
                # 保留最新的数据，丢弃旧数据
                self.send_queue = self.send_queue[-9:]
                print("警告: 发送队列已满，丢弃旧数据")
                
            # 将数据添加到队列
            self.send_queue.append({
                "class_id": mapped_class_id,
                "x_low": x_low,
                "x_high": x_high,
                "y_low": y_low,
                "y_high": y_high,
                "timestamp": time.time(),
                "orig_x": center_x,
                "orig_y": center_y,
                "orig_class": class_id,
                "retry": 0,  # 初始重试次数
            })
        return True
    
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
            # 4. 组装数据包
            data = bytes([
                data_to_send["class_id"],
                data_to_send["x_low"],
                data_to_send["x_high"],
                data_to_send["y_low"],
                data_to_send["y_high"]
            ])
    
            # 5. 发送数据
            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            self.last_stm32_send_time = current_time
    
            if DEBUG_WINDOW:
                print("\n----- 串口发送详细数据 [DEBUG] -----")
                print(f"十六进制数据: {' '.join([f'0x{b:02X}' for b in data])}")
    
                # 添加更直观的原始数据包展示
                print("原始数据包结构:")
                print(f"  [0] 0x{data[0]:02X} - 类别ID ({data_to_send['orig_class']} -> {data_to_send['class_id']})")
                print(f"  [1] 0x{data[1]:02X} - X坐标低8位")
                print(f"  [2] 0x{data[2]:02X} - X坐标高8位")
                print(f"  [3] 0x{data[3]:02X} - Y坐标低8位")
                print(f"  [4] 0x{data[4]:02X} - Y坐标高8位")
                print(f"数据包总长度: {len(data)} 字节，实际写入: {bytes_written} 字节")
                print(f"原始分类ID: {data_to_send['orig_class']} (十进制) -> {data_to_send['class_id']} (发送值)")
                print(f"原始X坐标: {data_to_send['orig_x']} -> 拆分: 低8位=0x{data_to_send['x_low']:02X}, 高8位=0x{data_to_send['x_high']:02X}")
                print(f"原始Y坐标: {data_to_send['orig_y']} -> 拆分: 低8位=0x{data_to_send['y_low']:02X}, 高8位=0x{data_to_send['y_high']:02X}")
                print(f"数据在队列中等待时间: {current_time - data_to_send['timestamp']:.3f}秒")
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


class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        # 更新为四大类
        self.class_names = {
            0: "厨余垃圾",
            1: "可回收垃圾",
            2: "有害垃圾",
            3: "其他垃圾",
        }
        # 为每个类别指定固定的颜色
        self.colors = {
            0: (86, 180, 233),  # 厨余垃圾 - 蓝色
            1: (230, 159, 0),  # 可回收垃圾 - 橙色
            2: (240, 39, 32),  # 有害垃圾 - 红色
            3: (0, 158, 115),  # 其他垃圾 - 绿色
        }
        self.serial_manager = SerialManager()
        self.waste_classifier = WasteClassifier()  # 初始化一次分类器

        # 添加检测节流相关变量
        self.last_detection_time = 0
        self.detection_interval = 0.5  # 每0.5秒最多发送一次数据
        self.last_detection_dict = {}  # 记录每个类别上次检测的位置和时间
        self.min_position_change = 20  # 位置变化阈值（像素）
        
        # 新增：顺序处理相关变量
        self.processing_queue = []  # 存储待处理的检测结果
        self.is_processing = False  # 是否正在处理队列
        self.send_interval = 1.0  # 发送间隔时间（秒），可配置
        self.process_thread = None  # 处理线程
        self.queue_lock = threading.Lock()  # 队列锁
        
        # 启动处理线程
        self.start_processing_thread()

    def start_processing_thread(self):
        """启动顺序处理线程"""
        self.is_processing = True
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("检测顺序处理线程已启动")
    
    def set_send_interval(self, interval):
        """设置发送间隔时间"""
        if interval > 0:
            self.send_interval = interval
            print(f"发送间隔时间已设置为 {interval} 秒")
    
    def _calculate_area(self, x1, y1, x2, y2):
        """计算检测框的面积"""
        return abs((x2 - x1) * (y2 - y1))
    
    def _process_queue(self):
        """处理队列中的检测结果"""
        while self.is_processing:
            detection_to_process = None
            
            with self.queue_lock:
                if self.processing_queue:
                    detection_to_process = self.processing_queue.pop(0)
            
            if detection_to_process:
                class_id, center_x, center_y, confidence, display_text, area = detection_to_process
                
                print(f"\n发送检测: {display_text}")
                print(f"置信度: {confidence:.2%}")
                print(f"中心点位置: ({center_x}, {center_y})")
                print(f"目标面积: {area} 像素^2")
                print("-" * 30)
                
                # 发送到串口
                self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                self.serial_manager.update_garbage_count(display_text)
                
                # 按照设定的间隔时间等待
                time.sleep(self.send_interval)
            else:
                # 队列为空时短暂休眠，避免CPU占用过高
                time.sleep(0.1)
    
    def _should_send_detection(self, class_id, center_x, center_y, confidence):
        """判断是否应该发送当前检测结果"""
        current_time = time.time()

        # 检查置信度
        if confidence < CONF_THRESHOLD:
            return False
            
        # 检查时间间隔
        if current_time - self.last_detection_time < self.detection_interval:
            # 时间间隔不足，检查是否有明显变化
            
            # 如果是新类别，则发送
            if class_id not in self.last_detection_dict:
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True
                
            # 检查位置变化
            last_pos = self.last_detection_dict[class_id]["position"]
            if (
                abs(center_x - last_pos[0]) > self.min_position_change
                or abs(center_y - last_pos[1]) > self.min_position_change
            ):
                # 位置有明显变化
                self.last_detection_dict[class_id] = {
                    "position": (center_x, center_y),
                    "time": current_time,
                }
                return True
                
            # 没有足够的变化，不发送
            return False
            
        # 时间间隔足够，更新状态并发送
        self.last_detection_time = current_time
        self.last_detection_dict[class_id] = {
            "position": (center_x, center_y),
            "time": current_time,
        }
        return True

    def detect(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD)
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # 收集所有检测到的对象
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                category_id, description = self.waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                
                # 计算面积
                area = self._calculate_area(x1, y1, x2, y2)
                
                # 将检测结果添加到列表
                detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': center_x,
                    'center_y': center_y,
                    'confidence': confidence,
                    'class_id': class_id,
                    'display_text': display_text,
                    'area': area
                })
                
                # 可视化（如果启用Debug窗口）
                if DEBUG_WINDOW:
                    color = self.colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{display_text} {confidence:.2f} A:{area}"
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
            
            # 按面积排序，从大到小
            detections.sort(key=lambda x: x['area'], reverse=True)
            
            # 将检测结果添加到处理队列
            with self.queue_lock:
                # 清空旧的队列
                self.processing_queue = []
                
                # 添加新的排序后的检测结果
                for detection in detections:
                    if self._should_send_detection(
                        detection['class_id'], 
                        detection['center_x'], 
                        detection['center_y'], 
                        detection['confidence']
                    ):
                        self.processing_queue.append((
                            detection['class_id'],
                            detection['center_x'],
                            detection['center_y'],
                            detection['confidence'],
                            detection['display_text'],
                            detection['area']
                        ))
                        
                        if DEBUG_WINDOW:
                            print(f"加入队列: {detection['display_text']}, 面积: {detection['area']}")
                    else:
                        print(f"检测到(不发送): {detection['display_text']}, 面积: {detection['area']}")
        
        return frame
    
    def cleanup(self):
        """清理资源"""
        print("正在清理YOLODetector资源...")
        self.is_processing = False
        
        # 等待处理线程结束
        if hasattr(self, "process_thread") and self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2.0)
                print("检测处理线程已终止")
            except Exception as e:
                print(f"终止处理线程出错: {str(e)}")
        
        # 清理串口管理器
        if hasattr(self, "serial_manager"):
            try:
                self.serial_manager.cleanup()
            except Exception as e:
                print(f"清理串口管理器出错: {str(e)}")


def create_detector(model_path):
    """
    创建YOLODetector实例
    Args:
        model_path: 模型文件路径
    Returns:
        detector: YOLODetector实例
    """
    import os

    # 检查文件是否存在

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    # 检查文件扩展名

    file_extension = os.path.splitext(model_path)[1].lower()
    if file_extension != ".pt":
        raise ValueError(f"不支持的模型格式: {file_extension}，仅支持 .pt 格式")
    print(f"使用 PyTorch 模型: {model_path}")
    try:
        return YOLODetector(model_path)
    except Exception as e:
        raise RuntimeError(f"加载 PyTorch 模型失败: {str(e)}")


def main():
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)

    # 使用新的创建检测器方法

    try:
        global model_path
        base_dir = get_script_directory()
        final_path = os.path.join(base_dir, model_path)
        detector = create_detector(final_path)
        detector.set_send_interval(2.0)
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
                window_name = "YOLO_detect"
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n程序正常退出")
                    break
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        # 清理资源

        if hasattr(detector, "serial_manager"):
            detector.serial_manager.cleanup()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
