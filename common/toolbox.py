import os
import torch
import cv2
import numpy as np

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "No GPU detected, will use CPU for inference"
    device_name = torch.cuda.get_device_name(0)
    return True, f"GPU enabled: {device_name}"

def find_ttyusb(max_index=10):
    """Find available ttyUSB devices
    
    Args:
        max_index: Maximum search index, default searches from ttyUSB0 to ttyUSB9
        
    Returns:
        Path string of existing ttyUSB device, returns None if not found
    """
    for index in range(max_index):
        ttyusb_path = f"/dev/ttyUSB{index}"
        if os.path.exists(ttyusb_path):
            print(f"Successfully found available ttyUSB device: {ttyusb_path}")
            return ttyusb_path
    print("Error: No available ttyUSB devices found")
    return None

def find_camera(width=1280, height=720):
    """Find available camera and set resolution"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Set the resolution explicitly
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify if the resolution was actually set
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"Successfully found available camera, index: {index}")
            print(f"Requested resolution: {width}x{height}, Actual resolution: {actual_width}x{actual_height}")
            
            return cap
        cap.release()
    print("Error: No available cameras found")
    return None

def crop_frame(frame, target_width=720, target_height=720, mode='center', points=None):
    """
    Crop video frame to specified size
    
    Args:
        frame: Input video frame (numpy array, OpenCV image format)
            Example: frame = cv2.imread('image.jpg') or frame from camera
            
        target_width: Target width, default 720
            Recommended values: Set according to model input requirements or display needs, common values: 224, 256, 320, 480, 640, 720
            Example: target_width=480 means cropped image width is 480 pixels
            
        target_height: Target height, default 720
            Recommended values: Set according to model input requirements or display needs, common values: 224, 256, 320, 480, 640, 720
            Example: target_height=480 means cropped image height is 480 pixels
            
        mode: Crop mode, options:
            - 'center': Crop from center (default)
                Use case: When target object is at image center, like frontal portraits, centered objects
                Usage: Keep center region, crop surrounding areas
                Example: crop_frame(frame, 480, 480, mode='center')
                Result: Extract 480x480 region from image center
                
            - 'left': Crop from left side
                Use case: When target object is on the left side of image
                Usage: Keep left region, crop right side
                Example: crop_frame(frame, 300, 720, mode='left')
                Result: Extract 300x720 region from left side, keeping left portion
                
            - 'right': Crop from right side
                Use case: When target object is on the right side of image
                Usage: Keep right region, crop left side
                Example: crop_frame(frame, 300, 720, mode='right')
                Result: Extract 300x720 region from right side, keeping right portion
                
            - 'top': Crop from top
                Use case: When target object is at top of image, like overhead view
                Usage: Keep top region, crop bottom
                Example: crop_frame(frame, 720, 300, mode='top')
                Result: Extract 720x300 region from top, keeping upper portion
                
            - 'bottom': Crop from bottom
                Use case: When target object is at bottom of image, like desktop objects, upward view
                Usage: Keep bottom region, crop top
                Example: crop_frame(frame, 720, 300, mode='bottom')
                Result: Extract 720x300 region from bottom, keeping lower portion
                
            Note: If original image size is smaller than target size, function will automatically adjust target size
            
        points: Specify crop region with top-left and bottom-right coordinates, format: [(x1, y1), (x2, y2)]
            When provided, will ignore target_width, target_height and mode parameters
            Example:
            points=[(100, 100), (500, 400)] 
            frame = crop_frame(frame, points=points) # Will crop rectangular region from (100,100) to (500,400)
    Returns:
        Cropped video frame
    """
    # Get original frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Direct crop by coordinate points
    if points is not None and len(points) == 2:
        # Top-left and bottom-right coordinates
        (start_x, start_y), (end_x, end_y) = points
        
        # Ensure coordinates are within valid range
        start_x = max(0, min(start_x, frame_width - 1))
        start_y = max(0, min(start_y, frame_height - 1))
        end_x = max(start_x + 1, min(end_x, frame_width))
        end_y = max(start_y + 1, min(end_y, frame_height))
        
        # Crop image
        return frame[start_y:end_y, start_x:end_x]
    
    # If original size is smaller than target size, adjust target size to smaller value
    if frame_width < target_width or frame_height < target_height:
        print(f"Warning: Original frame size({frame_width}x{frame_height}) is smaller than target size({target_width}x{target_height}), will adjust target size")
        # Adjust target size to minimum of original and target sizes
        target_width = min(frame_width, target_width)
        target_height = min(frame_height, target_height)
    
    # Calculate crop region
    if mode == 'center':
        # Crop from center
        start_x = (frame_width - target_width) // 2
        start_y = (frame_height - target_height) // 2
    elif mode == 'left':
        # Crop from left
        start_x = 0
        start_y = (frame_height - target_height) // 2
    elif mode == 'right':
        # Crop from right
        start_x = frame_width - target_width
        start_y = (frame_height - target_height) // 2
    elif mode == 'top':
        # Crop from top
        start_x = (frame_width - target_width) // 2
        start_y = 0
    elif mode == 'bottom':
        # Crop from bottom
        start_x = (frame_width - target_width) // 2
        start_y = frame_height - target_height
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")
    
    # Ensure starting coordinates are not negative
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    
    # Ensure not exceeding image boundaries
    if start_x + target_width > frame_width:
        start_x = frame_width - target_width
    if start_y + target_height > frame_height:
        start_y = frame_height - target_height
    
    # Crop image
    cropped_frame = frame[start_y:start_y+target_height, start_x:start_x+target_width]
    
    return cropped_frame



def get_script_directory():
    script_path = os.path.abspath(__file__)
    directory = os.path.dirname(script_path)
    print(f"Script directory: {directory}")
    return directory


class WasteClassifier:
    def __init__(self):
        # Classification names
        self.class_names = {
            0: "food waste",
            1: "recyclable trash",
            2: "harmful garbage",
            3: "Other garbage",
        }
        self.category_mapping = None
        # Classification descriptions (optional)
        self.category_descriptions = {
            0: "food waste",
            1: "recyclable trash",
            2: "harmful garbage",
            3: "Other garbage",
        }

    def get_category_info(self, class_id):
        """
        Get classification information for given class ID
        Returns: (category name, category description)
        """
        category_name = self.class_names.get(class_id, "Unknown classification")
        description = self.category_descriptions.get(class_id, "Unknown description")

        return category_name, description

    def print_classification(self, class_id):
        """Print classification information"""
        category_name, description = self.get_category_info(class_id)
        print(f"\nWaste classification information:")
        print(f"Category: {category_name}")
        print(f"Description: {description}")
        print("-" * 30)

        return f"{category_name}"
