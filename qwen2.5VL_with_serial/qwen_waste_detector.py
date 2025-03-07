import cv2
import torch
import serial
import threading
import time
import subprocess
import sys
import os
import json
import base64
from openai import OpenAI
from toolbox import get_script_directory, setup_gpu, find_camera, crop_frame
from toolbox import WasteClassifier

# 全局控制变量
DEBUG_WINDOW = True  # 是否显示调试窗口
ENABLE_SERIAL = True  # 是否启用串口通信
CONF_THRESHOLD = 0.6  # 置信度阈值
API_CALL_INTERVAL = 0.5  # API调用间隔（秒）

ENABLE_CROP = True  # 是否启用帧裁剪
X_CROP = 720  # 裁剪后的宽度
Y_CROP = 720  # 裁剪后的高度

# 串口配置
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_SERIAL_VALUE = 255


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
            # 4. 组装数据包
            data = bytes(
                [
                    data_to_send["class_id"],
                    data_to_send["x"],
                    data_to_send["y"],
                ]
            )

            # 5. 发送数据
            bytes_written = self.stm32_port.write(data)
            self.stm32_port.flush()
            self.last_stm32_send_time = current_time

            if DEBUG_WINDOW:
                print("\n----- 串口发送详细数据 [DEBUG] -----")
                print(f"十六进制数据: {' '.join([f'0x{b:02X}' for b in data])}")
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


class QwenDetector:
    def __init__(self, api_key=None):
        """初始化Qwen2.5-VL检测器"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen2.5-vl-72b-instruct"
        self.waste_classifier = WasteClassifier()
        self.serial_manager = SerialManager()
        
        # 检测控制参数
        self.last_detection_time = 0
        self.detection_interval = API_CALL_INTERVAL  # 控制API调用频率
        self.last_detection_dict = {}
        self.min_position_change = 20
        
        # 类别颜色映射
        self.colors = {
            0: (86, 180, 233),   # 厨余垃圾 - 蓝色
            1: (230, 159, 0),    # 可回收垃圾 - 橙色
            2: (240, 39, 32),    # 有害垃圾 - 红色
            3: (0, 158, 115),    # 其他垃圾 - 绿色
        }
        
        # 保存当前检测结果，用于在帧之间显示
        self.current_detections = []
        self.is_detection_in_progress = False
        
        print(f"Qwen2.5-VL检测器已初始化，使用模型: {self.model}")

    def _should_send_detection(self, class_id, center_x, center_y, confidence):
        """判断是否应该发送当前检测结果"""
        current_time = time.time()
        
        # 检查置信度
        if confidence < CONF_THRESHOLD:
            return False
            
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
            
        # 检查时间间隔
        last_detection_time = self.last_detection_dict[class_id]["time"]
        if current_time - last_detection_time > 3.0:  # 如果超过3秒没有更新，也发送
            self.last_detection_dict[class_id] = {
                "position": (center_x, center_y),
                "time": current_time,
            }
            return True
            
        return False

    def detect(self, frame):
        """使用Qwen2.5-VL API检测垃圾"""
        current_time = time.time()
        original_frame = frame.copy()
        
        # 始终绘制当前的检测结果（即使没有新的API调用）
        if DEBUG_WINDOW:
            frame = self.draw_detections(frame)
        
        # 控制API调用频率
        if current_time - self.last_detection_time < self.detection_interval or self.is_detection_in_progress:
            return frame
        
        # 标记检测开始
        self.is_detection_in_progress = True
        self.last_detection_time = current_time
        
        # 启动一个线程进行API调用，避免阻塞主线程
        detection_thread = threading.Thread(
            target=self._perform_detection, 
            args=(original_frame,)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        return frame
    
    def _perform_detection(self, frame):
        """在单独的线程中执行API检测"""
        try:
            # 转换图像为base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # 准备API请求
            prompt = (
                "请检测图像中的垃圾。将每个垃圾对象标识为以下四类之一(TIP:所有蔬菜/水果都算作厨余垃圾)："
                "0: 厨余垃圾, 1: 可回收垃圾, 2: 有害垃圾, 3: 其他垃圾。"
                "对于每个检测到的物体，提供类别ID、置信度和边界框坐标（左上和右下角以及中心点）。"
                "以JSON格式返回结果，格式如下："
                "```json\n"
                "{\n"
                "  \"detections\": [\n"
                "    {\n"
                "      \"class_id\": 0,\n"
                "      \"confidence\": 0.95,\n"
                "      \"x1\": 100,\n"
                "      \"y1\": 200,\n"
                "      \"x2\": 300,\n"
                "      \"y2\": 400,\n"
                "      \"center_x\": 200,\n"
                "      \"center_y\": 300\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
                "请仅返回JSON数据，不要添加任何其他说明。"
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            
            # 提取API响应文本
            response_text = response.choices[0].message.content
            
            # 尝试从响应中解析JSON
            try:
                # 查找JSON模式
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 尝试寻找花括号中的内容
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = response_text
                
                # 清理以确保有效的JSON
                json_str = json_str.strip()
                if json_str.startswith("```") and json_str.endswith("```"):
                    json_str = json_str[3:-3].strip()
                
                result = json.loads(json_str)
                
                # 清空旧的检测结果
                self.current_detections = []
                
                # 处理检测到的对象
                if 'detections' in result and len(result['detections']) > 0:
                    for det in result['detections']:
                        class_id = int(det.get('class_id', 0))
                        confidence = float(det.get('confidence', 0.0))
                        
                        # 获取坐标
                        if 'center_x' in det and 'center_y' in det:
                            center_x = int(det['center_x'])
                            center_y = int(det['center_y'])
                        else:
                            # 计算中心点
                            x1 = int(det.get('x1', 0))
                            y1 = int(det.get('y1', 0))
                            x2 = int(det.get('x2', 0))
                            y2 = int(det.get('y2', 0))
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                        
                        # 获取边界框坐标
                        if all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                            x1 = int(det['x1'])
                            y1 = int(det['y1'])
                            x2 = int(det['x2'])
                            y2 = int(det['y2'])
                        else:
                            # 如果没有提供边界框，创建一个围绕中心点的框
                            box_size = 100
                            x1 = max(0, center_x - box_size // 2)
                            y1 = max(0, center_y - box_size // 2)
                            x2 = min(frame.shape[1], center_x + box_size // 2)
                            y2 = min(frame.shape[0], center_y + box_size // 2)
                        
                        # 获取垃圾分类信息
                        category_id, description = self.waste_classifier.get_category_info(class_id)
                        display_text = f"{category_id}({description})"
                        
                        # 检查是否应该发送此检测
                        if self._should_send_detection(class_id, center_x, center_y, confidence):
                            print(f"\n发送检测: {display_text}")
                            print(f"置信度: {confidence:.2%}")
                            print(f"中心点位置: ({center_x}, {center_y})")
                            print("-" * 30)
                            
                            # 发送到串口
                            self.serial_manager.send_to_stm32(class_id, center_x, center_y)
                            self.serial_manager.update_garbage_count(display_text)
                        
                        # 保存检测结果以供绘制
                        self.current_detections.append({
                            'class_id': class_id,
                            'confidence': confidence,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'center_x': center_x,
                            'center_y': center_y,
                            'display_text': display_text
                        })
                
                else:
                    print("未检测到任何垃圾对象")
                    
            except json.JSONDecodeError as je:
                print(f"解析API响应中的JSON时出错: {je}")
                print(f"响应文本: {response_text}")
                
        except Exception as e:
            print(f"API请求错误: {str(e)}")
        
        finally:
            # 标记检测完成
            self.is_detection_in_progress = False
    
    def draw_detections(self, frame):
        """在帧上绘制当前的检测结果"""
        if not self.current_detections:
            return frame
            
        for det in self.current_detections:
            class_id = det['class_id']
            color = self.colors.get(class_id, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(frame, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
            
            # 绘制中心点
            cv2.circle(frame, (det['center_x'], det['center_y']), 5, (0, 255, 0), -1)
            
            # 绘制标签
            label = f"{det['display_text']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (det['x1'], det['y1'] - th - 10), (det['x1'] + tw + 10, det['y1']), color, -1)
            cv2.putText(frame, label, (det['x1'] + 5, det['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


def main():
    # 初始化检测器
    api_key = os.getenv("DASHSCOPE_API_KEY")  # 可以改成 api_key='sk-6e3df2f11a764c0698e7e96a6721e7e5' 这样的结构,将api-key硬编码入代码中
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量，请确保已正确设置API密钥")
        print("可以通过 export DASHSCOPE_API_KEY=your_key 设置环境变量")
        return
        
    detector = QwenDetector(api_key=api_key)
    
    # 查找摄像头
    cap = find_camera()
    if not cap:
        return
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print(f"- 调试窗口: {'开启' if DEBUG_WINDOW else '关闭'}")
    print(f"- 串口输出: {'开启' if ENABLE_SERIAL else '关闭'}")
    print(f"- API调用间隔: {API_CALL_INTERVAL}秒")
    print(f"- 帧裁剪: {'开启 ({X_CROP}x{Y_CROP})' if ENABLE_CROP else '关闭'}")
    print("- 按 'q' 键退出程序")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 如果启用了裁剪功能，则裁剪帧
            if ENABLE_CROP:
                frame = crop_frame(frame, target_width=X_CROP, target_height=Y_CROP, mode='center')
                
            # 处理当前帧
            processed_frame = detector.detect(frame)
            
            if DEBUG_WINDOW:
                window_name = "Qwen_VL_detect"
                cv2.imshow(window_name, processed_frame)
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
