from rknn.api import RKNN
import os
import torch
from ultralytics import YOLO
import numpy as np

def export_pt_to_onnx(pt_model_path, onnx_path):
    """将PT模型导出为ONNX格式"""
    print(f"Converting {pt_model_path} to ONNX format...")
    
    try:
        # 使用正确的YOLO命令行语法
        from ultralytics import YOLO
        model = YOLO(pt_model_path)
        model.export(format='onnx', opset=12, simplify=True)
        
        # 重命名导出的模型
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
    """将ONNX模型转换为RKNN格式"""
    print(f"Converting ONNX to RKNN format for {target_platform}...")
    
    # 初始化RKNN对象
    rknn = RKNN(verbose=True)
    
    # RKNN模型配置
    print('=> Config RKNN model')
    ret = rknn.config(mean_values=[[0, 0, 0]], 
                     std_values=[[255, 255, 255]],
                     target_platform=target_platform,
                     quantized_dtype="w16a16i_dfp",
                     quantized_algorithm="normal",
                     optimization_level=3)
    
    # 加载ONNX模型
    print('=> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed')
        return False
    
    # RKNN模型构建
    print('=> Building RKNN model')
    ret = rknn.build(do_quantization=True, 
                    dataset='./dataset.txt')  # 确保准备好量化数据集
    if ret != 0:
        print('Build RKNN model failed')
        return False
    
    # 导出RKNN模型
    print('=> Export RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed')
        return False
    
    print(f'Successfully exported RKNN model to {rknn_path}')
    return True

def prepare_quantization_dataset(dataset_txt_path='./dataset.txt', num_images=100):
    """准备用于量化的数据集列表"""
    print('Preparing quantization dataset...')
    
    # 确保训练集图片目录存在
    train_images_dir = 'train/images'
    if not os.path.exists(train_images_dir):
        print(f"Error: Training images directory {train_images_dir} not found")
        return False
        
    # 获取所有训练图片
    image_files = [f for f in os.listdir(train_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and ' ' not in f]
    
    if len(image_files) == 0:
        print("Error: No images found in training directory without spaces")
        return False
    
    # 选择指定数量的图片用于量化
    selected_images = image_files[:min(num_images, len(image_files))]
    
    # 获取绝对路径
    abs_train_dir = os.path.abspath(train_images_dir)
    
    # 创建数据集列表文件
    with open(dataset_txt_path, 'w') as f:
        for image in selected_images:
            abs_image_path = os.path.join(abs_train_dir, image)
            if os.path.exists(abs_image_path) and os.path.isfile(abs_image_path):  # 验证文件存在且是文件
                f.write(abs_image_path + '\n')
                print(f"Added to dataset: {abs_image_path}")  # 调试信息
            else:
                print(f"Skipped invalid file: {abs_image_path}")  # 调试信息
    
    print(f'Created quantization dataset list with {len(selected_images)} images')
    return True

def main():
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath(os.getcwd())
    
    # 配置路径
    pt_model_path = os.path.join(current_dir, 'runs', 'train', 'weights', 'best.pt')
    onnx_path = os.path.join(current_dir, 'model.onnx')
    rknn_path = os.path.join(current_dir, 'model.rknn')
    
    try:
        # 1. 准备量化数据集
        if not prepare_quantization_dataset():
            print("Failed to prepare quantization dataset")
            return
        
        # 2. PT转ONNX
        if not export_pt_to_onnx(pt_model_path, onnx_path):
            print("Failed to convert PT to ONNX")
            return
        
        # 3. ONNX转RKNN
        if not convert_onnx_to_rknn(onnx_path, rknn_path):
            print("Failed to convert ONNX to RKNN")
            return
        
        print("Model conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
    
    finally:
        # 清理中间文件
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists('./dataset.txt'):
            os.remove('./dataset.txt')

if __name__ == '__main__':
    main()