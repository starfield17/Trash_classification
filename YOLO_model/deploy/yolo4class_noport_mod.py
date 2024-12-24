import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import subprocess
import sys

# Global control variables
DEBUG_WINDOW = True
CONF_THRESHOLD = 0.9  # Confidence threshold
CAMERA_WIDTH = 1280   # Camera width
CAMERA_HEIGHT = 720   # Camera height

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "No GPU detected, using CPU for inference"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"GPU enabled: {device_name}"

class WasteClassifier:
    def __init__(self):
        # Classification names
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }
        
        # Category descriptions
        self.category_descriptions = {
            0: "厨余垃圾",
            1: "可回收利用垃圾",
            2: "有害垃圾",
            3: "其他垃圾"
        }

    def get_category_info(self, class_id):
        """
        Get classification info for given class ID
        Returns: (category_name, description)
        """
        category_name = self.class_names.get(class_id, "未知分类")
        description = self.category_descriptions.get(class_id, "未知描述")
        
        return category_name, description

    def print_classification(self, class_id):
        """Print classification info"""
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

        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }

        # Fixed colors for each class
        self.colors = {
            0: (86, 180, 233),    # Kitchen waste - Blue
            1: (230, 159, 0),     # Recyclable - Orange
            2: (240, 39, 32),     # Hazardous - Red
            3: (0, 158, 115)      # Other - Green
        }

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
        
        return frame

class ONNXDetector:
    def __init__(self, model_path):
        import onnxruntime as ort

        # 初始化运行时环境
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取模型信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 处理输入尺寸
        model_inputs = self.session.get_inputs()[0]
        input_shape = model_inputs.shape
        
        # 处理动态批次大小
        if input_shape[0] == 'batch' or input_shape[0] == -1:
            self.batch_size = 1
        else:
            self.batch_size = int(input_shape[0])
            
        # 设置输入尺寸（处理动态尺寸情况）
        if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
            self.input_height = 640
            self.input_width = 640
        else:
            self.input_height = int(input_shape[2])
            self.input_width = int(input_shape[3])
        
        print(f"Model input shape: batch={self.batch_size}, height={self.input_height}, width={self.input_width}")

        # 分类信息
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }

        # 颜色映射
        self.colors = {
            0: (86, 180, 233),    # 厨余垃圾 - 蓝色
            1: (230, 159, 0),     # 可回收垃圾 - 橙色
            2: (240, 39, 32),     # 有害垃圾 - 红色
            3: (0, 158, 115)      # 其他垃圾 - 绿色
        }

        # 检测历史记录
        self.detection_history = []
        self.history_length = 3  # 需要连续检测的帧数

    def preprocess(self, frame):
        """预处理输入图像"""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame input")

        try:
            # 调整图像尺寸
            target_size = (int(self.input_width), int(self.input_height))
            img = cv2.resize(frame, target_size)
            
            # 颜色空间转换和归一化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # 调整维度顺序并添加批次维度
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            
            return img
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            print(f"Frame shape: {frame.shape}")
            print(f"Target size: {target_size}")
            raise

    def is_valid_box(self, box, frame_shape):
        """验证检测框是否有效"""
        x1, y1, x2, y2 = box
        height, width = frame_shape[:2]
        
        # 检查坐标是否在图像范围内
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        
        # 检查框的最小尺寸
        min_size = 20
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return False
        
        # 检查宽高比是否合理
        if (y2 - y1) == 0:  # 防止除零
            return False
            
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return False
        
        return True

    def is_detection_stable(self, box, class_id, confidence):
        """检查检测是否稳定"""
        self.detection_history.append((box, class_id, confidence))
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)
            
        if len(self.detection_history) < self.history_length:
            return False
            
        # 检查类别稳定性
        recent_classes = [h[1] for h in self.detection_history]
        if not all(c == recent_classes[0] for c in recent_classes):
            return False
            
        # 检查位置稳定性
        recent_boxes = [h[0] for h in self.detection_history]
        box_centers = [((box[0] + box[2])/2, (box[1] + box[3])/2) for box in recent_boxes]
        
        for i in range(1, len(box_centers)):
            prev_center = box_centers[i-1]
            curr_center = box_centers[i]
            distance = ((curr_center[0] - prev_center[0])**2 + 
                       (curr_center[1] - prev_center[1])**2)**0.5
            if distance > 50:  # 50像素的位移阈值
                return False
                
        return True

    def postprocess(self, outputs, orig_shape):
        """后处理模型输出"""
        predictions = np.squeeze(outputs[0])
        
        if len(predictions.shape) == 1:
            return [], [], []

        # 计算置信度分数
        conf_scores = predictions[:, 4]
        class_scores = predictions[:, 5:]
        
        # 对类别分数应用sigmoid激活
        class_scores = 1 / (1 + np.exp(-class_scores))
        
        # 获取最高概率的类别及其分数
        class_ids = np.argmax(class_scores, axis=1)
        max_class_scores = np.max(class_scores, axis=1)
        
        # 计算最终置信度
        scores = conf_scores * max_class_scores
        
        # 应用置信度阈值
        mask = scores >= CONF_THRESHOLD
        
        if not np.any(mask):
            return [], [], []
        
        # 提取有效的检测结果
        boxes = predictions[mask, :4]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # 转换边界框格式
        boxes_xy = boxes.copy()
        boxes_xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 缩放到原始图像尺寸
        scale_x = orig_shape[1] / self.input_width
        scale_y = orig_shape[0] / self.input_height
        
        boxes_xy[:, [0, 2]] *= scale_x
        boxes_xy[:, [1, 3]] *= scale_y
        
        return boxes_xy.astype(np.int32), scores, class_ids.astype(np.int32)

    def detect(self, frame):
        """执行检测"""
        # 获取原始图像尺寸
        orig_shape = frame.shape
        
        # 预处理
        input_tensor = self.preprocess(frame)
        
        # 执行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        boxes, scores, class_ids = self.postprocess(outputs, orig_shape)
        
        # 处理检测结果
        if len(boxes) > 0:
            # 获取最高置信度的检测结果
            max_conf_idx = np.argmax(scores)
            box = boxes[max_conf_idx]
            
            # 验证检测框
            if not self.is_valid_box(box, frame.shape):
                return frame
            
            confidence = scores[max_conf_idx]
            class_id = class_ids[max_conf_idx]
            
            # 检查检测稳定性
            if not self.is_detection_stable(box, class_id, confidence):
                return frame
            
            # 计算中心点
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 获取分类信息
            waste_classifier = WasteClassifier()
            category_id, description = waste_classifier.get_category_info(class_id)
            display_text = f"{category_id}({description})"
            
            # 获取显示颜色
            color = self.colors.get(class_id, (255, 255, 255))
            
            # 绘制检测结果
            if DEBUG_WINDOW:
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 绘制中心点
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # 添加文本标签
                confidence = min(confidence, 1.0)
                label = f"{display_text} {confidence:.2%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 打印检测信息
            print(f"检测到物体:")
            print(f"置信度: {min(confidence, 1.0):.2%}")
            print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
            print(f"中心点位置: ({center_x}, {center_y})")
            print("-" * 30)
        
        return frame

def create_detector(model_path):
    """
    Create detector based on model file extension
    """
    import os
    
    file_extension = os.path.splitext(model_path)[1].lower()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if file_extension == '.pt':
        print(f"Using PyTorch model: {model_path}")
        try:
            return YOLODetector(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
            
    elif file_extension == '.onnx':
        print(f"Using ONNX model: {model_path}")
        try:
            import importlib
            if importlib.util.find_spec("onnxruntime") is None:
                raise ImportError("Please install onnxruntime: pip install onnxruntime-gpu or pip install onnxruntime")
            return ONNXDetector(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
            
    else:
        raise ValueError(f"Unsupported model format: {file_extension}, only .pt or .onnx supported")

def find_camera():
    """Find available camera"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Found available camera at index: {index}")
            return cap
        cap.release()
    
    print("Error: No available camera found")
    return None

def main():
    use_gpu, device_info = setup_gpu()
    print("\nDevice information:")
    print(device_info)
    print("-" * 30)
    
    try:
        model_path = 'best.onnx'  # or 'best.onnx'
        detector = create_detector(model_path)
    except Exception as e:
        print(f"Failed to create detector: {str(e)}")
        return
    
    cap = find_camera()
    if not cap:
        return
    
    if DEBUG_WINDOW:
        window_name = 'YOLOv8 Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\nSystem startup:")
    print("- Camera ready")
    print(f"- Debug window: {'enabled' if DEBUG_WINDOW else 'disabled'}")
    print("- Press 'q' to exit")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break
            
            frame = detector.detect(frame)
            
            if DEBUG_WINDOW:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProgram exited normally")
                    break
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, exiting")
    finally:
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
