import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import subprocess
import sys

# Global control variables
DEBUG_WINDOW = False
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

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }

        self.colors = {
            0: (86, 180, 233),    # Kitchen waste - Blue
            1: (230, 159, 0),     # Recyclable - Orange
            2: (240, 39, 32),     # Hazardous - Red
            3: (0, 158, 115)      # Other - Green
        }

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img

    def postprocess(self, outputs, orig_shape):
        predictions = np.squeeze(outputs[0])
        
        if len(predictions.shape) == 1:
            return [], [], []

        mask = predictions[:, 4] >= CONF_THRESHOLD
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return [], [], []
            
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        scores = np.max(predictions[:, 5:], axis=1) * predictions[:, 4]
        boxes = predictions[:, :4]
        
        scale_x = orig_shape[1] / self.input_width
        scale_y = orig_shape[0] / self.input_height
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        return boxes.astype(np.int32), scores, class_ids.astype(np.int32)

    def detect(self, frame):
        orig_shape = frame.shape
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        boxes, scores, class_ids = self.postprocess(outputs, orig_shape)
        
        if len(boxes) > 0:
            max_conf_idx = np.argmax(scores)
            box = boxes[max_conf_idx]
            confidence = scores[max_conf_idx]
            class_id = class_ids[max_conf_idx]
            
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
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
        model_path = 'best.pt'  # or 'best.onnx'
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
