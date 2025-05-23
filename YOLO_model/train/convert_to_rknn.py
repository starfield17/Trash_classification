from rknn.api import RKNN
import os
import torch
from ultralytics import YOLO
import numpy as np

def export_pt_to_onnx(pt_model_path, onnx_path):
    """Convert PT model to ONNX format"""
    print(f"Converting {pt_model_path} to ONNX format...")
    
    try:
        # Use correct YOLO command syntax
        from ultralytics import YOLO
        model = YOLO(pt_model_path)
        model.export(format='onnx', opset=12, simplify=True)
        
        # Rename the exported model
        default_onnx = pt_model_path.replace('.pt', '.onnx')
        if os.path.exists(default_onnx):
            os.rename(default_onnx, onnx_path)
            print(f"Successfully exported ONNX model to {onnx_path}")
            return True
        else:
            print("ONNX file not found after export")
            return False
    except Exception as e:
        print(f"Export failed: {str(e)}")
        return False

def convert_onnx_to_rknn(onnx_path, rknn_path, target_platform='rk3588'):
    """Convert ONNX model to RKNN format"""
    print(f"Converting ONNX to RKNN format for {target_platform}...")
    
    # Initialize RKNN object
    rknn = RKNN(verbose=True)
    
    # RKNN model configuration
    print('=> Configuring RKNN model')
    ret = rknn.config(mean_values=[[0, 0, 0]], 
                     std_values=[[255, 255, 255]],
                     target_platform=target_platform,
                     quantized_dtype="w16a16i_dfp",
                     quantized_algorithm="normal",
                     optimization_level=3)
    
    # Load ONNX model
    print('=> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Failed to load ONNX model')
        return False
    
    # Build RKNN model
    print('=> Building RKNN model')
    ret = rknn.build(do_quantization=True, 
                    dataset='./dataset.txt')  # Ensure quantization dataset is prepared
    if ret != 0:
        print('Failed to build RKNN model')
        return False
    
    # Export RKNN model
    print('=> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Failed to export RKNN model')
        return False
    
    print(f'Successfully exported RKNN model to {rknn_path}')
    return True

def prepare_quantization_dataset(dataset_txt_path='./dataset.txt', num_images=100):
    """Prepare dataset list for quantization"""
    print('Preparing quantization dataset...')
    
    # Ensure training images directory exists
    train_images_dir = 'train/images'
    if not os.path.exists(train_images_dir):
        print(f"Error: Training images directory {train_images_dir} not found")
        return False
        
    # Get all training images
    image_files = [f for f in os.listdir(train_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and ' ' not in f]
    
    if len(image_files) == 0:
        print("Error: No images found in training directory without spaces")
        return False
    
    # Select specified number of images for quantization
    selected_images = image_files[:min(num_images, len(image_files))]
    
    # Get absolute paths
    abs_train_dir = os.path.abspath(train_images_dir)
    
    # Create dataset list file
    with open(dataset_txt_path, 'w') as f:
        for image in selected_images:
            abs_image_path = os.path.join(abs_train_dir, image)
            if os.path.exists(abs_image_path) and os.path.isfile(abs_image_path):  # Verify file exists and is a file
                f.write(abs_image_path + '\n')
                print(f"Added to dataset: {abs_image_path}")  # Debug info
            else:
                print(f"Skipped invalid file: {abs_image_path}")  # Debug info
    
    print(f'Created quantization dataset list with {len(selected_images)} images')
    return True

def main():
    # Get absolute path of current working directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Configure paths
    pt_model_path = os.path.join(current_dir, 'runs', 'train', 'weights', 'best.pt')
    onnx_path = os.path.join(current_dir, 'model.onnx')
    rknn_path = os.path.join(current_dir, 'model.rknn')
    
    try:
        # 1. Prepare quantization dataset
        if not prepare_quantization_dataset():
            print("Failed to prepare quantization dataset")
            return
        
        # 2. Convert PT to ONNX
        if not export_pt_to_onnx(pt_model_path, onnx_path):
            print("Failed to convert PT to ONNX")
            return
        
        # 3. Convert ONNX to RKNN
        if not convert_onnx_to_rknn(onnx_path, rknn_path):
            print("Failed to convert ONNX to RKNN")
            return
        
        print("Model conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
    
    finally:
        # Clean up intermediate files
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists('./dataset.txt'):
            os.remove('./dataset.txt')

if __name__ == '__main__':
    main()
