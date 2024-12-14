import cv2
import torch
from ultralytics import YOLO
import numpy as np
import sys

# 全局控制变量
DEBUG_WINDOW = False
CONF_THRESHOLD = 0.9  # 置信度阈值

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

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
            7: 1,  # inner_packing -> 可回收垃圾
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

    def detect(self, frame):
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
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                # 使用WasteClassifier获取分类信息
                waste_classifier = WasteClassifier()
                display_text = waste_classifier.print_classification(class_id)
                
                color = self.colors.get(class_id, (255, 255, 255))
                
                # 在PC上始终显示检测框和信息
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
    
    window_name = 'YOLOv8垃圾分类检测'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print("- 按 'q' 键退出程序")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            frame = detector.detect(frame)
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n程序正常退出")
                break
            
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
