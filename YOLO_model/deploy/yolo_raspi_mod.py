import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import pigpio
import subprocess
import sys
# 全局控制变量
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.5  # 置信度阈值

# 串口配置
STM32_PORT = '/dev/ttyS0'  # STM32串口
STM32_BAUD = 9600
SCREEN_TX = 23  # GPIO23 作为TX
SCREEN_RX = 24  # GPIO24 作为RX
SCREEN_BAUD = 9600


# 串口通信协议
FRAME_HEADER = b'\xff\xff'  # 帧头(STM32)
FRAME_FOOTER = b'\xff\xff'  # 帧尾(STM32)
SCREEN_END = b'\xff\xff\xff'  # 串口屏结束符
def check_pigpiod():
    """检查pigpiod守护进程是否运行"""
    try:
        # 尝试运行pigpiod
        subprocess.run(['sudo', 'pigpiod'], capture_output=True)
    except Exception as e:
        print(f"警告: 启动pigpiod失败: {str(e)}")
        print("请手动运行: sudo pigpiod")
        sys.exit(1)
        
class SoftwareSerial:
    def __init__(self, tx_pin, rx_pin, baud_rate):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("无法连接到pigpio守护进程")
        
        self.tx_pin = tx_pin
        self.rx_pin = rx_pin
        self.baud_rate = baud_rate
        self.is_open = False
        
        # 配置GPIO
        self.pi.set_mode(tx_pin, pigpio.OUTPUT)
        self.pi.set_mode(rx_pin, pigpio.INPUT)
        
        # 创建软串口
        self.serial = self.pi.serial_open(rx_pin, baud_rate, 8)
        self.is_open = True
        
        # 用于数据接收的缓冲区
        self._read_buffer = bytearray()
    
    def write(self, data):
        if self.is_open:
            try:
                self.pi.wave_clear()
                self.pi.wave_add_serial(self.tx_pin, self.baud_rate, data)
                wave_id = self.pi.wave_create()
                if wave_id >= 0:
                    self.pi.wave_send_once(wave_id)
                    while self.pi.wave_tx_busy():  # 等待发送完成
                        time.sleep(0.001)
                    self.pi.wave_delete(wave_id)
            except Exception as e:
                print(f"软串口发送数据失败: {str(e)}")
    
    def close(self):
        if self.is_open:
            try:
                self.pi.serial_close(self.serial)
                self.pi.stop()
                self.is_open = False
            except Exception as e:
                print(f"关闭软串口失败: {str(e)}")


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
        
        # 初始化STM32串口
        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(STM32_PORT, STM32_BAUD, timeout=0.1)
                print(f"STM32串口已初始化: {STM32_PORT}")
            except Exception as e:
                print(f"STM32串口初始化失败: {str(e)}")
                self.stm32_port = None
        
        # 初始化软串口屏
        try:
            self.screen_port = SoftwareSerial(SCREEN_TX, SCREEN_RX, SCREEN_BAUD)
            print(f"串口屏已初始化: GPIO{SCREEN_TX}(TX), GPIO{SCREEN_RX}(RX)")
        except Exception as e:
            print(f"串口屏初始化失败: {str(e)}")
            self.screen_port = None
        
        # 启动STM32数据接收线程
        if self.stm32_port:
            self.receive_thread = threading.Thread(target=self.receive_stm32_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
    
    def send_to_screen(self, text, encoding="GB2312"):
        """发送数据到串口屏"""
        if self.screen_port and self.screen_port.is_open:
            try:
                # 发送文本命令
                command = f't0.txt=\"{text}\"'.encode(encoding)
                self.screen_port.write(command)
                time.sleep(0.01)  # 短暂延时确保命令发送完成
                # 发送结束符
                self.screen_port.write(SCREEN_END)
                print(f"串口屏输出: {text}")
            except Exception as e:
                print(f"串口屏输出失败: {str(e)}")


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
        # 使用YOLOv8进行检测
        results = self.model(frame, conf=CONF_THRESHOLD)
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                class_name = self.class_names.get(class_id, "unknown")
                color = self.colors.get(class_id, (255, 255, 255))
                
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                print(f"\n检测到物体:")
                print(f"类别: {class_name}")
                print(f"置信度: {confidence:.2%}")
                print(f"位置: ({x1}, {y1}), ({x2}, {y2})")
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
    check_pigpiod()
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
