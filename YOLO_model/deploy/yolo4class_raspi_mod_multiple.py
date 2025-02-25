import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import subprocess
import sys

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
STM32_PORT = '/dev/ttyS0'  # 选择使用的串口
STM32_BAUD = 115200
CAMERA_WIDTH = 1280   # 摄像头宽度
CAMERA_HEIGHT = 720   # 摄像头高度
MAX_SERIAL_VALUE = 255  # 串口发送的最大值


def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

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
                    STM32_PORT, 
                    STM32_BAUD, 
                    timeout=0.1,
                    write_timeout=0.1
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
        if (garbage_type != self.current_detection or 
            (current_time - self.detection_lost_time > self.DETECTION_RESET_TIME and 
             self.detection_lost_time > 0)):
            # 重置检测状态
            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        
        # 如果已经达到稳定识别时间
        if (current_time - self.detection_start_time >= self.STABILITY_THRESHOLD and 
            not self.stable_detection):
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
        self.detected_items.append({
            'count': self.garbage_count,
            'type': garbage_type,
            'quantity': 1,
            'status': "正确"
        })
        
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
        """处理队列中的一批数据"""
        if not self.stm32_port or not self.stm32_port.is_open:
            return
        
        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return
        
        # 获取一批数据进行处理
        batch_to_send = None
        with self.queue_lock:
            if self.send_queue:
                # 每次最多取出5个批次数据处理
                batch_to_send = self.send_queue.pop(0)
        
        if not batch_to_send:
            return
        
        try:
            # 短暂延时确保设备准备就绪
            self.stm32_port.reset_input_buffer()
            self.stm32_port.reset_output_buffer()
            time.sleep(0.01)
            
            print("\n----- 串口发送批量数据 -----")
            print(f"批次ID: {batch_to_send['batch_id']}")
            print(f"目标数量: {len(batch_to_send['detections'])}")
            print(f"队列等待时间: {current_time - batch_to_send['timestamp']:.3f}秒")
            
            for idx, detection in enumerate(batch_to_send['detections']):
                class_id, center_x, center_y = detection
                
                # 类别ID映射处理
                if class_id == 0:
                    mapped_class_id = self.zero_mapping
                else:
                    mapped_class_id = class_id
                
                # 确保所有值在有效范围内
                mapped_class_id = min(255, max(0, mapped_class_id))
                x_scaled = min(MAX_SERIAL_VALUE, max(0, int(center_x * MAX_SERIAL_VALUE / CAMERA_WIDTH)))
                y_scaled = min(MAX_SERIAL_VALUE, max(0, int(center_y * MAX_SERIAL_VALUE / CAMERA_HEIGHT)))
                
                # 组装并发送数据包
                data = bytes([mapped_class_id, x_scaled, y_scaled])
                bytes_written = self.stm32_port.write(data)
                self.stm32_port.flush()
                
                # 每个数据包之间短暂延时
                time.sleep(0.01)
                
                # 打印每个目标的详细信息
                print(f"\n目标 {idx + 1}/{len(batch_to_send['detections'])}:")
                print(f"发送的原始数据: {' '.join([f'0x{b:02X}' for b in data])}")
                print(f"实际写入字节数: {bytes_written}")
                print(f"分类ID: {class_id} -> {mapped_class_id if class_id == 0 else class_id}")
                print(f"原始坐标: X={center_x}, Y={center_y}")
                print(f"缩放后坐标: X={x_scaled} (0x{x_scaled:02X}), Y={y_scaled} (0x{y_scaled:02X})")
            
            self.last_stm32_send_time = current_time
            print(f"\n批次发送完成，总耗时: {time.time() - current_time:.3f}秒")
            
            with self.queue_lock:
                print(f"队列中剩余批次: {len(self.send_queue)}")
            
            print("-" * 30)
            
        except serial.SerialTimeoutException:
            print("串口写入超时，将重新入队...")
            # 超时的批次重新放回队列头部
            with self.queue_lock:
                # 增加重试次数
                retry_count = batch_to_send.get('retry', 0) + 1
                if retry_count <= 3:  # 最多重试3次
                    batch_to_send['retry'] = retry_count
                    # 重新放入队列头部
                    self.send_queue.insert(0, batch_to_send)
                    print(f"批次将重试发送，第{retry_count}次尝试")
                else:
                    print(f"批次重试次数已达上限，丢弃该批次")
                    
        except Exception as e:
            print(f"串口发送错误: {str(e)}")
            # 其他错误时也考虑重试
            with self.queue_lock:
                retry_count = batch_to_send.get('retry', 0) + 1
                if retry_count <= 3:
                    batch_to_send['retry'] = retry_count
                    self.send_queue.insert(0, batch_to_send)
                    print(f"错误后重试，第{retry_count}次尝试")

    def send_to_stm32(self, detections=None):
        """将检测结果添加到发送队列"""
        if not self.stm32_port or not self.stm32_port.is_open:
            return False
        
        if detections is None:
            detections = self.current_detections
        
        if not detections:
            return False  # 没有检测结果，直接返回
        
        # 创建批次数据
        batch_data = {
            'batch_id': int(time.time() * 1000),  # 使用时间戳作为批次ID
            'timestamp': time.time(),
            'detections': detections.copy(),  # 复制检测结果避免引用问题
            'retry': 0  # 初始重试次数为0
        }
        
        # 添加到发送队列
        with self.queue_lock:
            if len(self.send_queue) >= self.MAX_QUEUE_SIZE:
                # 队列满时，保留最新的数据，移除较早的数据
                excess = len(self.send_queue) - self.MAX_QUEUE_SIZE + 1
                self.send_queue = self.send_queue[excess:]
                print(f"警告: 发送队列已满，丢弃{excess}个较早的批次")
            
            self.send_queue.append(batch_data)
            queue_size = len(self.send_queue)
        
        print(f"批次已加入队列，当前队列长度: {queue_size}")
        return True

    def cleanup(self):
        """清理资源"""
        self.is_running = False
        print("正在清理资源...")
        
        # 等待队列处理线程结束
        if hasattr(self, 'queue_thread') and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            print("队列处理线程已终止")
        
        # 关闭串口
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            print("串口已关闭")
            
class WasteClassifier:
    def __init__(self):
        # 分类名称
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }
        
        # 细分类到大分类的映射 - 移除,因为我们直接使用四大类
        self.category_mapping = None
        
        # 分类名称对应的描述(可选)
        self.category_descriptions = {
            0: "厨余垃圾",
            1: "可回收利用垃圾",
            2: "有害垃圾",
            3: "其他垃圾"
        }

    def get_category_info(self, class_id):
        """
        获取给定类别ID的分类信息
        返回: (分类名称, 分类描述)
        """
        category_name = self.class_names.get(class_id, "未知分类")
        description = self.category_descriptions.get(class_id, "未知描述")
        
        return category_name, description

    def print_classification(self, class_id):
        """打印分类信息"""
        category_name, description = self.get_category_info(class_id)
        print(f"\n垃圾分类信息:")
        print(f"分类类别: {category_name}")
        print(f"分类说明: {description}")
        print("-" * 30)
        
        return f"{category_name}"
class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        # 更新为四大类
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }
        # 为每个类别指定固定的颜色
        self.colors = {
            0: (86, 180, 233),    # 厨余垃圾 - 蓝色
            1: (230, 159, 0),     # 可回收垃圾 - 橙色
            2: (240, 39, 32),     # 有害垃圾 - 红色
            3: (0, 158, 115)      # 其他垃圾 - 绿色
        }
        self.serial_manager = SerialManager()
      
    def detect(self, frame):
        self.serial_manager.clear_detections()
        results = self.model(frame, conf=CONF_THRESHOLD)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                waste_classifier = WasteClassifier()
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                
                color = self.colors.get(class_id, (255, 255, 255))
                
                self.serial_manager.add_detection(class_id, center_x, center_y)
                
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{display_text} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                print(f"\n检测到物体 {len(self.serial_manager.current_detections)}:")
                print(f"类别: {display_text}")
                print(f"置信度: {confidence:.2%}")
                print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
                print(f"中心点位置: ({center_x}, {center_y})")
                print("-" * 30)
            
            # 发送所有检测结果
            self.serial_manager.send_to_stm32()
            
            # 更新垃圾计数
            for box in boxes:
                class_id = int(box.cls[0].item())
                waste_classifier = WasteClassifier()
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                self.serial_manager.update_garbage_count(display_text)
        return frame

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
    if file_extension != '.pt':
        raise ValueError(f"不支持的模型格式: {file_extension}，仅支持 .pt 格式")
    
    print(f"使用 PyTorch 模型: {model_path}")
    try:
        return YOLODetector(model_path)
    except Exception as e:
        raise RuntimeError(f"加载 PyTorch 模型失败: {str(e)}")
    
def find_camera():
    """查找可用的摄像头"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功找到可用摄像头，索引为: {index}")
            return cap
        cap.release()
    
    print("错误: 未找到任何可用的摄像头")
    return None

def main():
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)
    
    # 使用新的创建检测器方法
    try:
        model_path = 'best.pt'  # 或 'best.onnx'
        detector = create_detector(model_path)
    except Exception as e:
        print(f"创建检测器失败: {str(e)}")
        return
    
    cap = find_camera()
    if not cap:
        return
    
    if DEBUG_WINDOW:
        window_name = 'YOLOv8检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
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
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n程序正常退出")
                    break
            
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        # 清理资源
        if hasattr(detector, 'serial_manager'):
            detector.serial_manager.cleanup()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
