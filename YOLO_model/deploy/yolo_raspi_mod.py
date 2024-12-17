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
STM32_PORT = '/dev/ttyS0'  # STM32串口(TX-> GPIO14,RX->GPIO15)
STM32_BAUD = 115200
SCREEN_PORT = '/dev/ttyAMA2'  # 串口屏串口(TX-GPIO0> ,RX->GPIO1)
SCREEN_BAUD = 9600

# 串口通信协议
FRAME_HEADER = b''  # 帧头(STM32)可为空
FRAME_FOOTER = b''  # 帧尾(STM32)可为空
#SCREEN_END = b'\xff\xff\xff'  # 串口屏结束符
SCREEN_END = bytes.fromhex('ff ff ff')
def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

class SerialManager:
    def __init__(self):
        self.stm32_port = None
        self.screen_port = None
        self.is_running = True
        self.last_stm32_send_time = 0
        self.last_screen_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1  # 最小发送间隔（秒）
        
        # 垃圾计数和记录相关
        self.garbage_count = 0  # 垃圾计数器
        self.detected_items = []  # 存储检测到的垃圾记录
        
        # 防重复计数和稳定性检测相关
        self.last_count_time = 0  # 上次计数的时间
        self.COUNT_COOLDOWN = 5.0  # 计数冷却时间（秒）
        self.is_counting_locked = False  # 计数锁定状态
        self.last_detected_type = None  # 上次检测到的垃圾类型
        
        # 稳定性检测相关(当前物体)
        self.current_detection = None  # 当前正在检测的物体类型
        self.detection_start_time = 0  # 开始检测的时间
        self.STABILITY_THRESHOLD = 1.0  # 稳定识别所需时间（秒）
        self.stable_detection = False  # 是否已经稳定识别
        self.detection_lost_time = 0  # 丢失检测的时间
        self.DETECTION_RESET_TIME = 0.5  # 检测重置时间（秒）

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
            except Exception as e:
                print(f"STM32串口初始化失败: {str(e)}")
                self.stm32_port = None

        # 初始化串口屏
            try:
                self.screen_port = serial.Serial(
                    SCREEN_PORT,
                    SCREEN_BAUD,
                    timeout=0.1,
                    write_timeout=0.1
                )
                print(f"串口屏已初始化: {SCREEN_PORT}")
                self.init_screen_table()  # 初始化表格
            except Exception as e:
                print(f"串口屏初始化失败: {str(e)}")
                self.screen_port = None

        # 启动数据接收线程
        if self.stm32_port:
            self.receive_thread = threading.Thread(target=self.receive_stm32_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()

    def receive_stm32_data(self):
        """接收STM32数据的线程函数"""
        buffer_size = 10240  # 设置合理的缓冲区大小
        
        while self.is_running and self.stm32_port and self.stm32_port.is_open:
            try:
                # 检查串口是否有数据可读
                if self.stm32_port.in_waiting > 0:
                    # 读取数据，限制读取大小
                    data = self.stm32_port.read(min(self.stm32_port.in_waiting, buffer_size))
                    
                    if data:
                        try:
                            # 尝试解码数据（去除无效字符）
                            decoded_data = data.decode('utf-8', errors='replace').strip()
                            # 过滤掉全是 null 字符或 0xFF 的数据
                            if any(c not in ['\x00', '\xff'] for c in decoded_data):
                                # 将数据转换为十六进制字符串
                                hex_data = ' '.join(f'0x{byte:02X}' for byte in data)
                                print(f"接收到的原始数据: {hex_data}")
                                print(f"解码后的数据: {decoded_data}")
                        except UnicodeDecodeError as e:
                            hex_data = ' '.join(f'0x{byte:02X}' for byte in data)
                            print(f"数据解码错误: {str(e)}")
                            print(f"原始数据: {hex_data}")
                    
                    # 清理缓冲区
                    self.stm32_port.reset_input_buffer()
                time.sleep(0.01)
                
            except serial.SerialException as e:
                print(f"串口通信错误: {str(e)}")
                # 尝试重新打开串口
                try:
                    if self.stm32_port.is_open:
                        self.stm32_port.close()
                    time.sleep(1)  # 等待一秒后重试
                    self.stm32_port.open()
                    print("串口重新打开成功")
                except Exception as reopen_error:
                    print(f"串口重新打开失败: {str(reopen_error)}")
                    break  # 如果重新打开失败，退出循环
                    
            except Exception as e:
                print(f"其他错误: {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                # 根据需要决定是否终止线程，以下示例选择继续运行
                # 如果需要终止，可以取消注释下一行
                # break
                    
        print("串口接收线程终止")

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
        """更新垃圾计数并更新显示"""
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
        
        # 更新显示
        self.update_screen_table()

    def init_screen_table(self):
        """初始化串口屏表格"""
        # 清空所有文本框
        for i in range(1, 9):  # 8行
            for j in range(1, 5):  # 4列
                self.send_to_screen_component(f"x{i}y{j}", "")
    
    def send_to_screen_component(self, component_id, text, encoding="UTF-8"):
        """发送数据到特定的串口屏组件"""
        if not self.screen_port or not self.screen_port.is_open:
            return

        try:
            command = f'{component_id}.txt=\"{text}\"'.encode(encoding)
            self.screen_port.write(command)
            time.sleep(0.01)
            self.screen_port.write(SCREEN_END)
            self.screen_port.flush()
            
        except Exception as e:
            print(f"串口屏组件 {component_id} 输出失败: {str(e)}")

    def update_screen_table(self):
        """更新串口屏表格显示"""
        # 确保只显示最近的8条记录
        recent_items = self.detected_items[-8:]
        
        # 先清空所有单元格
        self.init_screen_table()
        
        # 更新显示内容
        for i, item in enumerate(recent_items, 1):
            # 序号
            self.send_to_screen_component(f"x{i}y1", str(item['count']))
            # 垃圾种类
            self.send_to_screen_component(f"x{i}y2", item['type'])
            # 数量
            self.send_to_screen_component(f"x{i}y3", str(item['quantity']))
            # 状态
            self.send_to_screen_component(f"x{i}y4", item['status'])

    def send_to_screen(self, text, encoding="UTF-8"):
        """保留原方法，但主要使用新的表格更新方法"""
        self.update_garbage_count(text)

    def send_to_stm32(self, class_id):
        """发送数据到STM32"""
        if not self.stm32_port or not self.stm32_port.is_open:
            return
    
        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return
    
        try:
            self.stm32_port.reset_input_buffer()
            self.stm32_port.reset_output_buffer()
            
            # 发送数据
            data = FRAME_HEADER + str(class_id).encode('utf-8') + FRAME_FOOTER
            self.stm32_port.write(data)
            self.stm32_port.flush()
            
            self.last_stm32_send_time = current_time
            print(f"发送数据: {data}")
            
        except Exception as e:
            print(f"串口发送错误: {str(e)}")

    def cleanup(self):
        """清理串口资源"""
        self.is_running = False
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
        if self.screen_port and self.screen_port.is_open:
            self.screen_port.close()
            
class WasteClassifier:
    def __init__(self):
        # 原始类别映射
        self.class_names = {
            0: 'potato',
            1: 'daikon',
            2: 'carrot',
            3: 'bottle',
            4: 'can',
            5: 'battery',
            6: 'drug',
            7: 'inner_packing',
            8: 'tile',
            9: 'stone',
            10: 'brick'
        }
        
        # 细分类到大分类的映射
        self.category_mapping = {
            0: 0,  # potato -> 厨余垃圾
            1: 0,  # daikon -> 厨余垃圾
            2: 0,  # carrot -> 厨余垃圾
            3: 1,  # bottle -> 可回收垃圾
            4: 1,  # can -> 可回收垃圾
            5: 2,  # battery -> 有害垃圾
            6: 2,  # drug -> 有害垃圾
            7: 2,  # inner_packing -> 有害垃圾
            8: 3,  # tile -> 其他垃圾
            9: 3,  # stone -> 其他垃圾
            10: 3  # brick -> 其他垃圾
        }
        
        # 大分类名称
        self.category_names = {
            0: "厨余垃圾",
            1: "可回收垃圾",
            2: "有害垃圾",
            3: "其他垃圾"
        }

    def get_category_info(self, class_id):
        """
        获取给定类别ID的详细分类信息
        返回: (细分类名称, 大分类ID, 大分类名称)
        """
        specific_name = self.class_names.get(class_id, "unknown")
        category_id = self.category_mapping.get(class_id)
        category_name = self.category_names.get(category_id, "未知分类")
        
        return specific_name, category_id, category_name

    def print_classification(self, class_id):
        """打印分类信息"""
        specific_name, category_id, category_name = self.get_category_info(class_id)
        print(f"\n垃圾分类信息:")
        print(f"具体物品: {specific_name}")
        print(f"所属分类: {category_name}")
        print("-" * 30)
        
        return f"{specific_name}({category_name})"
        
class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.class_names = {
            0: 'potato',
            1: 'daikon',
            2: 'carrot',
            3: 'bottle',
            4: 'can',
            5: 'battery',
            6: 'drug',
            7: 'inner_packing',
            8: 'tile',
            9: 'stone',
            10: 'brick'
        }

        self.colors = {}
        np.random.seed(42)
        for class_id in self.class_names:
            self.colors[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))

        self.serial_manager = SerialManager()

    def detect(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) > 0:
                confidences = [box.conf[0].item() for box in boxes]
                max_conf_idx = np.argmax(confidences)
                box = boxes[max_conf_idx]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                waste_classifier = WasteClassifier()
                specific_name, category_id, category_name = waste_classifier.get_category_info(class_id)
                display_text = f"{specific_name}({category_name})"
                
                color = self.colors.get(class_id, (255, 255, 255))
                
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{display_text} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                print(f"检测到物体:")
                print(f"置信度: {confidence:.2%}")
                print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
                print(f"中心点位置: ({center_x}, {center_y})")
                print("-" * 30)
                #self.serial_manager.send_to_stm32(class_id)
                self.serial_manager.send_to_stm32(category_id)
                self.serial_manager.send_to_screen(display_text)
        
        return frame

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
    
    detector = YOLODetector(
        model_path='best.pt'
    )
    
    cap = find_camera()
    if not cap:
        return
    
    if DEBUG_WINDOW:
        window_name = 'YOLOv8&v11检测'
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
