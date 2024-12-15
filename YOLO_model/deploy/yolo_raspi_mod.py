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
STM32_BAUD = 9600
SCREEN_PORT = '/dev/ttyAMA2'  # 串口屏串口(TX-GPIO0> ,RX->GPIO1)
SCREEN_BAUD = 9600

# 串口通信协议
FRAME_HEADER = b''  # 帧头(STM32)可为空
FRAME_FOOTER = b''  # 帧尾(STM32)可为空
SCREEN_END = b'\xff\xff\xff'  # 串口屏结束符 等同于SCREEN_END = bytes.fromhex('ff ff ff')
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

        # 初始化STM32串口
        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(
                    STM32_PORT, 
                    STM32_BAUD, 
                    timeout=0.1,
                    write_timeout=0.1  # 添加写超时
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
                write_timeout=0.1  # 添加写超时
            )
            print(f"串口屏已初始化: {SCREEN_PORT}")
        except Exception as e:
            print(f"串口屏初始化失败: {str(e)}")
            self.screen_port = None

        # 若STM32串口初始化成功，则启动数据接收线程
        if self.stm32_port:
            self.receive_thread = threading.Thread(target=self.receive_stm32_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()

    def receive_stm32_data(self):
        """接收STM32数据的线程函数"""
        while self.is_running and self.stm32_port and self.stm32_port.is_open:
            data = self.stm32_port.read_all()
            if data:
                print(f"从STM32收到数据: {data}")
            time.sleep(0.1)

    def send_to_screen(self, text, encoding="GB2312"):
        """发送数据到串口屏，添加了时间间隔控制和缓冲区清理"""
        if not self.screen_port or not self.screen_port.is_open:
            return

        current_time = time.time()
        if current_time - self.last_screen_send_time < self.MIN_SEND_INTERVAL:
            return  # 如果距离上次发送时间太短，直接返回

        try:
            # 清空缓冲区
            self.screen_port.reset_input_buffer()
            self.screen_port.reset_output_buffer()
            
            # 构建并发送命令
            command = f't0.txt=\"{text}\"'.encode(encoding)
            self.screen_port.write(command)
            time.sleep(0.01)  # 串口屏需要的延时
            self.screen_port.write(SCREEN_END)
            
            # 等待数据发送完成
            self.screen_port.flush()
            
            self.last_screen_send_time = current_time
            print(f"串口屏输出: {text}")
            
        except serial.SerialTimeoutException:
            print("串口屏发送超时")
        except Exception as e:
            print(f"串口屏输出失败: {str(e)}")

    def send_to_stm32(self, class_id):
        """发送数据到STM32，添加了时间间隔控制和缓冲区清理"""
        if not self.stm32_port or not self.stm32_port.is_open:
            return

        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return  # 如果距离上次发送时间太短，直接返回

        try:
            # 清空接收缓冲区
            self.stm32_port.reset_input_buffer()
            # 清空发送缓冲区
            self.stm32_port.reset_output_buffer()
            
            # 构建完整的数据帧
            data = FRAME_HEADER + str(class_id).encode('utf-8') + FRAME_FOOTER
            self.stm32_port.write(data)
            # 等待数据发送完成
            self.stm32_port.flush()
            
            self.last_stm32_send_time = current_time
            print(f"发送数据: {data}")
            
        except serial.SerialTimeoutException:
            print("串口发送超时")
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
