import cv2
import numpy as np
import torch
import json

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

class GarbageDetectorPyTorch:
    def __init__(self, model_path, labels_path, num_threads, enable_display=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_display = enable_display
        torch.set_num_threads(num_threads)
        # 加载标签映射
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # 加载PyTorch模型
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # 获取模型输入尺寸
        self.IMG_SIZE = 224  # 根据模型要求设置
        
        # 定义类别及其对应颜色
        self.categories = {
            '其他垃圾': (128, 128, 128),
            '厨余垃圾': (0, 255, 0),
            '可回收物': (0, 0, 255),
            '有害垃圾': (255, 0, 0)
        }

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

    def detect(self, frame):
        height, width = frame.shape[:2]
        
        # 定义中心区域
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x1 = max(center_x - box_size, 0)
        y1 = max(center_y - box_size, 0)
        x2 = min(center_x + box_size, width)
        y2 = min(center_y + box_size, height)
        
        # 提取中心区域并进行预处理
        roi = frame[y1:y2, x1:x2]
        input_data = self.preprocess_image(roi)
        
        # 运行推理
        with torch.no_grad():
            output_data = self.model(input_data)
            probabilities = torch.nn.functional.softmax(output_data[0], dim=0)
            class_id = torch.argmax(probabilities).item()
            confidence = probabilities[class_id].item()
        
        # 获取标签和类别
        label = self.labels.get(str(class_id), "未知类别")
        category = self.get_category(label)
        
        # 可视化处理
        if confidence > 0.9 and self.enable_display:
            color = self.categories.get(category, (0, 255, 0))
            
            # 绘制中心检测区域
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            padding = 10
            
            # 准备显示文本
            text_category = f"类别: {category}"
            text_item = f"物品: {label.split('/')[-1]}"
            text_conf = f"置信度: {confidence:.1%}"
            
            # 计算文本位置
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
            
        if confidence > 0.9:
            print("\n检测结果:")
            print(f"类别: {category}")
            print(f"物品: {label.split('/')[-1]}")
            print(f"置信度: {confidence:.1%}")
            print("-" * 30)
        
        return frame

def find_camera():
    for index in range(101):  # 搜索 0-100 号摄像头
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功找到可用摄像头，索引为: {index}")
            return cap
        cap.release()  # 释放不可用的摄像头
    
    print("错误: 未找到任何可用的摄像头")
    return None

def main(enable_display=True):
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)

    detector = GarbageDetectorPyTorch(
        model_path='garbage_classifier_ResNet18.pt',
        labels_path='garbage_classify_rule.json',
        num_threads=8,
        enable_display=enable_display
    )
    
    cap = find_camera()
    
    if enable_display:
        window_name = '垃圾分类检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    if enable_display:
        print("- 按 'q' 键退出程序")
        print("- 将物品放置在画面中心区域")
    else:
        print("- 按 Ctrl+C 退出程序")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            frame = detector.detect(frame)
            
            if enable_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n程序正常退出")
                    break
                
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        cap.release()
        if enable_display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # 设置 enable_display=False 可以关闭窗口显示
    main(enable_display=False)
