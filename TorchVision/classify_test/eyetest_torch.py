import cv2
import numpy as np
import torch
import json

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "GPU not detected, will use CPU for inference"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"GPU enabled: {device_name}"

class GarbageDetectorPyTorch:
    def __init__(self, model_path, labels_path, num_threads, enable_display=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_display = enable_display
        torch.set_num_threads(num_threads)
        
        # Load label mapping
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # Load PyTorch model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Get model input size
        self.IMG_SIZE = 224  # Set according to model requirements
        
        # Define categories and their corresponding colors
        self.categories = {
            'Other Waste': (128, 128, 128),
            'Kitchen Waste': (0, 255, 0),
            'Recyclables': (0, 0, 255),
            'Hazardous Waste': (255, 0, 0)
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
        
        # Define center detection region
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x1 = max(center_x - box_size, 0)
        y1 = max(center_y - box_size, 0)
        x2 = min(center_x + box_size, width)
        y2 = min(center_y + box_size, height)
        
        # Extract center region and preprocess
        roi = frame[y1:y2, x1:x2]
        input_data = self.preprocess_image(roi)
        
        # Run inference
        with torch.no_grad():
            output_data = self.model(input_data)
            probabilities = torch.nn.functional.softmax(output_data[0], dim=0)
            class_id = torch.argmax(probabilities).item()
            confidence = probabilities[class_id].item()
        
        # Get label and category
        label = self.labels.get(str(class_id), "Unknown Category")
        category = self.get_category(label)
        
        # Visualization processing
        if confidence > 0.9 and self.enable_display:
            color = self.categories.get(category, (0, 255, 0))
            
            # Draw center detection region
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Set text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            padding = 10
            
            # Prepare display text
            text_category = f"Category: {category}"
            text_item = f"Item: {label.split('/')[-1]}"
            text_conf = f"Confidence: {confidence:.1%}"
            
            # Calculate text positions
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
            print("\nDetection Result:")
            print(f"Category: {category}")
            print(f"Item: {label.split('/')[-1]}")
            print(f"Confidence: {confidence:.1%}")
            print("-" * 30)
        
        return frame

def find_camera():
    for index in range(101):  # Search cameras 0-100
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Successfully found available camera, index: {index}")
            return cap
        cap.release()  # Release unavailable cameras
    
    print("Error: No available cameras found")
    return None

def main(enable_display=True):
    use_gpu, device_info = setup_gpu()
    print("\nDevice Information:")
    print(device_info)
    print("-" * 30)

    detector = GarbageDetectorPyTorch(
        model_path='garbage_classifier.pt',
        labels_path='garbage_classify_rule.json',
        num_threads=8,
        enable_display=enable_display
    )
    
    cap = find_camera()
    
    if enable_display:
        window_name = 'Garbage Classification Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\nSystem Started:")
    print("- Camera ready")
    if enable_display:
        print("- Press 'q' to exit")
        print("- Place items in the center region")
    else:
        print("- Press Ctrl+C to exit")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break
            
            frame = detector.detect(frame)
            
            if enable_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProgram exited normally")
                    break
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, program exiting")
    finally:
        cap.release()
        if enable_display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Set enable_display=False to disable window display
    main(enable_display=False)
