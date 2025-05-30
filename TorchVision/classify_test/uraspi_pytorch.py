import cv2
import numpy as np
import torch
import json
import serial

# 全局控制变量
DEBUG_WINDOW = False
ENABLE_SERIAL = True
THREADS=4
# 串口配置
SERIAL_PORT = '/dev/ttyS0'
SERIAL_BAUD = 9600

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

class GarbageDetectorPyTorch:
    def __init__(self, model_path, labels_path, num_threads):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(num_threads)
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        self.IMG_SIZE = 224
        
        self.categories = {
            '其他垃圾': {'color': (128, 128, 128), 'code': '0'},
            '厨余垃圾': {'color': (0, 255, 0), 'code': '1'},
            '可回收物': {'color': (0, 0, 255), 'code': '2'},
            '有害垃圾': {'color': (255, 0, 0), 'code': '3'}
        }
        
        self.serial_port = None
        if ENABLE_SERIAL:
            try:
                self.serial_port = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
                print(f"串口已初始化: {SERIAL_PORT}")
            except Exception as e:
                print(f"串口初始化失败: {str(e)}")
                self.serial_port = None

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).to(self.device)
        return img

    def get_category(self, label):
        for category in self.categories.keys():
            if label.startswith(category):
                return category
        return None

    def send_serial_data(self, category):
        if ENABLE_SERIAL and self.serial_port and self.serial_port.is_open:
            try:
                code = self.categories[category]['code']
                self.serial_port.write(code.encode())
                print(f"串口输出: {code}")
            except Exception as e:
                print(f"串口输出失败: {str(e)}")

    def detect(self, frame):
        height, width = frame.shape[:2]
        
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x1 = max(center_x - box_size, 0)
        y1 = max(center_y - box_size, 0)
        x2 = min(center_x + box_size, width)
        y2 = min(center_y + box_size, height)
        
        roi = frame[y1:y2, x1:x2]
        input_data = self.preprocess_image(roi)
        
        with torch.no_grad():
            output_data = self.model(input_data)
            probabilities = torch.nn.functional.softmax(output_data[0], dim=0)
            class_id = torch.argmax(probabilities).item()
            confidence = probabilities[class_id].item()
        
        label = self.labels.get(str(class_id), "未知类别")
        category = self.get_category(label)
        
        if confidence > 0.5:
            if DEBUG_WINDOW:
                color = self.categories[category]['color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                padding = 10
                
                text_category = f"类别: {category}"
                text_item = f"物品: {label.split('/')[-1]}"
                text_conf = f"置信度: {confidence:.1%}"
                
                y_offset = y1 - padding
                for text in [text_conf, text_item, text_category]:
                    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    y_offset -= h + padding
                    
                    cv2.rectangle(frame, 
                                (x1, y_offset - padding), 
                                (x1 + w + padding * 2, y_offset + h + padding),
                                (0, 0, 0),
                                -1)
                    
                    cv2.putText(frame, text, 
                              (x1 + padding, y_offset + h),
                              font, font_scale, color, thickness)
            
            print("\n检测结果:")
            print(f"类别: {category}")
            print(f"物品: {label.split('/')[-1]}")
            print(f"置信度: {confidence:.1%}")
            print("-" * 30)
            
            self.send_serial_data(category)
        
        return frame

def find_camera():
    for index in range(101):
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

    detector = GarbageDetectorPyTorch(
        model_path='garbage_classifier.pt',
        labels_path='garbage_classify_rule.json',
        num_threads=THREADS
    )
    
    cap = find_camera()
    if not cap:
        return
    
    if DEBUG_WINDOW:
        window_name = '垃圾分类检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print(f"- 调试窗口: {'开启' if DEBUG_WINDOW else '关闭'}")
    print(f"- 串口输出: {'开启' if ENABLE_SERIAL else '关闭'}")
    print("- 按 'q' 键退出程序")
    print("- 将物品放置在画面中心区域")
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
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()
        if detector.serial_port:
            detector.serial_port.close()

if __name__ == '__main__':
    main()
