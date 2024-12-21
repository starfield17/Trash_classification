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

# 串口通信协议
FRAME_HEADER = b''  # 帧头(STM32)可为空
FRAME_FOOTER = b''  # 帧尾(STM32)可为空
FULL_SIGNAL = "123"  # 定义满载信号

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
                                if decoded_data == FULL_SIGNAL:
                                    print("检测到满载信号")
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
                    
        print("串口接收线程终止")

def send_to_stm32(self, class_id, max_retries=3, retry_delay=0.1):
    """
    发送数据到STM32，带有重试机制和更好的错误处理
    
    Args:
        class_id: 要发送的分类ID
        max_retries: 最大重试次数
        retry_delay: 重试间隔时间(秒)
    """
    if not self.stm32_port or not self.stm32_port.is_open:
        print("串口未开启或未连接")
        return False

    # 检查发送间隔
    current_time = time.time()
    if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
        return False

    # 数据验证
    try:
        class_id = int(class_id)
        if not 0 <= class_id <= 3:  # 确保class_id在有效范围内
            print(f"无效的分类ID: {class_id}")
            return False
    except (ValueError, TypeError):
        print(f"分类ID格式错误: {class_id}")
        return False

    # 准备发送数据
    data = FRAME_HEADER + str(class_id).encode('utf-8') + FRAME_FOOTER
    
    # 重试循环
    for attempt in range(max_retries):
        try:
            # 检查串口状态
            if not self.stm32_port.is_open:
                print("串口已关闭，尝试重新打开")
                self.stm32_port.open()
                
            # 只在第一次尝试时重置输出缓冲区
            if attempt == 0:
                self.stm32_port.reset_output_buffer()

            # 发送数据
            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            
            # 验证发送的数据长度
            if bytes_written != len(data):
                raise serial.SerialException(f"数据发送不完整: {bytes_written}/{len(data)} 字节")
            
            self.last_stm32_send_time = current_time
            print(f"发送数据成功: {data}, 字节数: {bytes_written}")
            return True
            
        except serial.SerialException as e:
            print(f"串口发送错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            try:
                self.stm32_port.close()
                time.sleep(retry_delay)
                self.stm32_port.open()
            except Exception as reopen_error:
                print(f"串口重新打开失败: {str(reopen_error)}")
                continue
                
        except Exception as e:
            print(f"其他错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
            
    print("发送数据失败，已达到最大重试次数")
    return False

    def cleanup(self):
        """清理串口资源"""
        self.is_running = False
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            
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
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                
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
                self.serial_manager.send_to_stm32(class_id)
        
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
