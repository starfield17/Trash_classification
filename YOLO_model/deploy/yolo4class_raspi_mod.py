import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import subprocess
import sys
import logging
# 全局控制变量
DEBUG_WINDOW = False
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9  # 置信度阈值
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 串口配置
STM32_PORT = '/dev/ttyAMA2'  # STM32串口(TX-> GPIO0,RX->GPIO1)
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
    def __init__(self,
                 port=STM32_PORT,
                 baudrate=STM32_BAUD,
                 timeout=0.1,
                 write_timeout=0.1,
                 frame_header=b'\xAA\xBB',
                 frame_footer=b'\xCC\xDD',
                 full_signal="123",
                 min_send_interval=0.1):
        """
        初始化串口管理器。

        Args:
            port (str): 串口端口名称。
            baudrate (int): 波特率。
            timeout (float): 读取超时时间（秒）。
            write_timeout (float): 写入超时时间（秒）。
            frame_header (bytes): 帧头。
            frame_footer (bytes): 帧尾。
            full_signal (str): 满载信号。
            min_send_interval (float): 最小发送间隔（秒）。
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.frame_header = frame_header
        self.frame_footer = frame_footer
        self.full_signal = full_signal
        self.min_send_interval = min_send_interval

        self.stm32_port = None
        self.is_running = True
        self.last_stm32_send_time = 0

        self.receive_buffer = bytearray()
        self.lock = threading.Lock()

        self.setup_serial()
        self.start_receive_thread()

    def setup_serial(self):
        """初始化串口连接。"""
        try:
            self.stm32_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout
            )
            logging.info(f"串口已初始化: {self.port} @ {self.baudrate}波特率")
        except serial.SerialException as e:
            logging.error(f"串口初始化失败: {e}")
            self.stm32_port = None

    def start_receive_thread(self):
        """启动接收线程。"""
        if self.stm32_port and self.stm32_port.is_open:
            self.receive_thread = threading.Thread(target=self.receive_stm32_data, daemon=True)
            self.receive_thread.start()
            logging.info("串口接收线程已启动")
        else:
            logging.warning("串口未打开，无法启动接收线程")

    def stop(self):
        """停止接收线程并关闭串口。"""
        self.is_running = False
        if self.receive_thread.is_alive():
            self.receive_thread.join()
            logging.info("串口接收线程已停止")
        if self.stm32_port and self.stm32_port.is_open:
            try:
                self.stm32_port.close()
                logging.info("串口已关闭")
            except serial.SerialException as e:
                logging.error(f"关闭串口失败: {e}")

    def calculate_checksum(self, data):
        """
        计算简单的校验和。

        Args:
            data (bytes): 数据字节。

        Returns:
            int: 校验和。
        """
        return sum(data) & 0xFF

    def receive_stm32_data(self):
        """接收STM32数据的线程函数，基于帧结构和校验和。"""
        while self.is_running and self.stm32_port and self.stm32_port.is_open:
            try:
                if self.stm32_port.in_waiting > 0:
                    data = self.stm32_port.read(self.stm32_port.in_waiting)
                    with self.lock:
                        self.receive_buffer.extend(data)
                        logging.debug(f"接收到的数据: {data.hex()}")

                        while True:
                            header_index = self.receive_buffer.find(self.frame_header)
                            if header_index == -1:
                                # 未找到帧头，清空缓冲区
                                self.receive_buffer.clear()
                                break
                            
                            if len(self.receive_buffer) < header_index + len(self.frame_header) + 2:
                                # 数据不足以包含最小帧（数据字节 + 校验和 + 帧尾）
                                break

                            footer_index = self.receive_buffer.find(self.frame_footer, header_index + len(self.frame_header))
                            if footer_index == -1:
                                # 未找到帧尾，等待更多数据
                                break

                            # 提取完整帧
                            frame = self.receive_buffer[header_index + len(self.frame_header):footer_index]
                            self.receive_buffer = self.receive_buffer[footer_index + len(self.frame_footer):]

                            if len(frame) < 2:
                                logging.warning("接收到的帧长度不足，忽略")
                                continue

                            data_byte = frame[0]
                            checksum = frame[1]

                            calculated_checksum = self.calculate_checksum(frame[:-1])
                            if checksum != calculated_checksum:
                                logging.warning("校验和错误，数据可能损坏")
                                logging.debug(f"接收到的校验和: {checksum}, 计算得到的校验和: {calculated_checksum}")
                                continue

                            try:
                                decoded_data = frame[:-1].decode('utf-8', errors='replace').strip()
                                logging.info(f"解码后的数据: {decoded_data}")
                                if decoded_data == self.full_signal:
                                    logging.info("检测到满载信号")
                            except UnicodeDecodeError as e:
                                logging.error(f"数据解码错误: {e}")
                                logging.debug(f"原始数据: {frame[:-1].hex()}")

                time.sleep(0.01)

            except serial.SerialException as e:
                logging.error(f"串口通信错误: {e}")
                self.handle_serial_exception()
            except Exception as e:
                logging.error(f"其他错误: {e}")
                logging.debug(f"错误类型: {type(e).__name__}")

        logging.info("串口接收线程终止")

    def handle_serial_exception(self):
        """处理串口通信错误，尝试重新连接。"""
        try:
            if self.stm32_port.is_open:
                self.stm32_port.close()
                logging.info("串口已关闭，准备重新打开")
            time.sleep(1)  # 等待一秒后重试
            self.stm32_port.open()
            logging.info("串口重新打开成功")
        except serial.SerialException as reopen_error:
            logging.error(f"串口重新打开失败: {reopen_error}")
            self.is_running = False  # 无法重新打开，停止线程

    def send_to_stm32(self, class_id, max_retries=3, retry_delay=0.1):
        """
        发送数据到STM32，带有重试机制和更好的错误处理。

        Args:
            class_id (int): 要发送的分类ID。
            max_retries (int): 最大重试次数。
            retry_delay (float): 重试间隔时间（秒）。

        Returns:
            bool: 发送是否成功。
        """
        if not self.stm32_port or not self.stm32_port.is_open:
            logging.error("串口未开启或未连接")
            return False

        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.min_send_interval:
            logging.debug("发送间隔过短，跳过发送")
            return False

        try:
            class_id = int(class_id)
            if not 0 <= class_id <= 255:
                logging.error(f"无效的分类ID: {class_id}")
                return False
        except (ValueError, TypeError):
            logging.error(f"分类ID格式错误: {class_id}")
            return False

        # 准备发送数据，包含帧头、数据、校验和和帧尾
        data_byte = class_id.to_bytes(1, 'little')
        checksum = self.calculate_checksum(data_byte)
        data = self.frame_header + data_byte + bytes([checksum]) + self.frame_footer

        for attempt in range(1, max_retries + 1):
            try:
                with self.lock:
                    bytes_written = self.stm32_port.write(data)
                    self.stm32_port.flush()
                    logging.debug(f"发送的数据: {data.hex()}, 字节数: {bytes_written}")

                if bytes_written != len(data):
                    raise serial.SerialException(f"数据发送不完整: {bytes_written}/{len(data)} 字节")

                self.last_stm32_send_time = current_time
                logging.info(f"发送数据成功: {data.hex()}, 字节数: {bytes_written}")
                return True

            except serial.SerialException as e:
                logging.error(f"串口发送错误 (尝试 {attempt}/{max_retries}): {e}")
                self.handle_serial_exception()
                time.sleep(retry_delay)
            except Exception as e:
                logging.error(f"其他错误 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(retry_delay)

        logging.error("发送数据失败，已达到最大重试次数")
        return False

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
        detector.serial_manager.stop()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
