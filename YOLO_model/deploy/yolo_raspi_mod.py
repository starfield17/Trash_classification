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
CONF_THRESHOLD = 0.5  # 置信度阈值

# 串口配置
STM32_PORT = '/dev/ttyS0'  # STM32串口(TX-> GPIO14,RX->GPIO15)
STM32_BAUD = 9600
SCREEN_PORT = '/dev/ttyAMA2'  # 串口屏串口(TX-GPIO0> ,RX->GPIO1)
SCREEN_BAUD = 9600

# 串口通信协议
FRAME_HEADER = b'\xff\xff'  # 帧头(STM32)
FRAME_FOOTER = b'\xff\xff'  # 帧尾(STM32)
SCREEN_END = b'\xff\xff\xff'  # 串口屏结束符

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


class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = YOLO(model_path)

        # 类别名称和颜色映射
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

        # 为每个类别生成随机颜色
        self.colors = {}
        np.random.seed(42)
        for class_id in self.class_names:
            self.colors[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))

        # 初始化串口管理器
        self.serial_manager = SerialManager()

    def detect(self, frame):
        """对单帧图像进行检测"""
        results = self.model(frame, conf=CONF_THRESHOLD)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # 只处理置信度最高的检测结果
            if len(boxes) > 0:
                # 获取最高置信度的检测框
                confidences = [box.conf[0].item() for box in boxes]
                max_conf_idx = np.argmax(confidences)
                box = boxes[max_conf_idx]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 计算中心点坐标
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.class_names.get(class_id, "unknown")
                color = self.colors.get(class_id, (255, 255, 255))
                
                if DEBUG_WINDOW:
                    # 画出边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # 画出中心点
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{class_name} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                print(f"\n检测到物体:")
                print(f"类别: {class_name}")
                print(f"置信度: {confidence:.2%}")
                print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
                print(f"中心点位置: ({center_x}, {center_y})")
                print("-" * 30)
                
                # 发送检测结果到STM32
                self.serial_manager.send_to_stm32(class_id)
                # 发送检测结果到串口屏
                self.serial_manager.send_to_screen(f"检测到: {class_name}")
        
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
