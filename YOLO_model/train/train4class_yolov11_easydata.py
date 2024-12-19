from ultralytics import YOLO
import yaml
import os
import json
from sklearn.model_selection import train_test_split
import shutil
import albumentations as A
import cv2
import numpy as np
from pathlib import Path

datapath='./label'  # 根据实际情况修改

def check_and_clean_dataset(data_dir):
    """检查数据集完整性并清理无效数据"""
    print("正在检查数据集完整性...")
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]
    valid_pairs = []
    print(f"找到 {len(image_files)} 张图片")
    # 检查每个图片是否有效且有对应的标签文件
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        if '.' in base_name:
            pass  # 如果需要，可以添加额外的处理逻辑
        json_file = os.path.join(data_dir, base_name + '.json')
        # 检查图片完整性
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 损坏或无效的图片文件: {img_file}")
                continue
                
            # 检查图片尺寸是否合理
            height, width = img.shape[:2]
            if height < 10 or width < 10:
                print(f"警告: 图片尺寸过小: {img_file}")
                continue
                
        except Exception as e:
            print(f"警告: 读取图片 {img_file} 时出错: {e}")
            continue
            
        # 检查标签文件是否存在且有效
        if os.path.exists(json_file):
            # 确保标签文件确实是一个 JSON 文件，而不是其他格式
            if not json_file.lower().endswith('.json'):
                print(f"警告: 标签文件扩展名不正确 (应为 .json): {json_file}")
                continue
            
            # 验证 JSON 文件内容
            if validate_json_file(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                        
                    # 验证标签数据结构
                    if 'labels' not in label_data:
                        print(f"警告: 标签文件结构无效 (缺少 'labels' 键): {json_file}")
                        continue
                        
                    valid_pairs.append(img_file)
                except Exception as e:
                    print(f"警告: 处理标签文件 {json_file} 时发生错误: {e}")
                    continue
            else:
                # JSON 文件无效，跳过
                continue
        else:
            print(f"警告: 找不到对应的标签文件: {json_file}")
    
    print(f"找到 {len(valid_pairs)} 对有效的图片和标签文件")
    return valid_pairs
    
def validate_json_file(json_path):
    """
    验证 JSON 文件是否有效。
    返回 True 表示有效，False 表示无效。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        print(f"警告: JSON 解码错误 - 无效的 JSON 文件: {json_path}")
        return False
    except UnicodeDecodeError:
        print(f"警告: Unicode 解码错误 - 无效的 JSON 文件: {json_path}")
        return False
    except Exception as e:
        print(f"警告: 无法读取 JSON 文件 {json_path} - 错误: {e}")
        return False


def create_data_yaml():
    """创建数据配置文件 - 使用四大分类"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        'path': current_dir,
        'train': os.path.join(current_dir, 'train/images'),
        'val': os.path.join(current_dir, 'val/images'),
        'test': os.path.join(current_dir, 'test/images'),
        'names': {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        },
        'nc': 4  # 改为4个大类
    }
    
    with open('data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
def prepare_dataset(data_dir, valid_pairs):
    """准备数据集 - 修改验证集划分比例"""
    # 确保验证集至少有10张图片
    if len(valid_pairs) < 15:
        raise ValueError(f"Not enough valid data pairs ({len(valid_pairs)}). Need at least 15 images.")
    
    # 清理现有目录
    for split in ['train', 'val', 'test']:
        if os.path.exists(split):
            shutil.rmtree(split)
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(split, subdir), exist_ok=True)
    
    # 数据集划分 (80% 训练, 10% 验证, 10% 测试)
    train_files, temp = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # 处理每个分割
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split...")
        for img_file in files:
            # 修改这部分代码
            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(data_dir, img_file)
            # 确保我们使用正确的JSON文件名
            src_json = os.path.join(data_dir, base_name + '.json')
            dst_img = os.path.join(split_name, 'images', img_file)
            dst_txt = os.path.join(split_name, 'labels', base_name + '.txt')
            
            # 复制图片和转换标签
            shutil.copy2(src_img, dst_img)
            convert_labels(src_json, dst_txt)
            
        print(f"{split_name}: {len(files)} images")
    
    return len(train_files), len(val_files), len(test_files)

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """转换边界框从x1,y1,x2,y2到YOLO格式"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # 计算中心点坐标和宽高
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # 确保值在0-1范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height
def convert_labels(json_file, txt_file):
    """转换标签文件 - 直接转换为四大类"""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False
            
        # 获取图片路径
        base_name = os.path.splitext(json_file)[0]
        possible_extensions = ['.jpg', '.jpeg', '.png']
        img_path = None
        
        for ext in possible_extensions:
            temp_path = base_name + ext
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None:
            print(f"Warning: No corresponding image file found for: {json_file}")
            return False
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image: {img_path}")
            return False
        
        img_height, img_width = img.shape[:2]
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 细分类到大分类的映射
        category_mapping = {
            # 厨余垃圾 (0)
            'potato': 0,
            'daikon': 0,
            'carrot': 0,
            # 可回收垃圾 (1)
            'bottle': 1,
            'can': 1,
            # 有害垃圾 (2)
            'battery': 2,
            'drug': 2,
            'inner_packing': 2,
            # 其他垃圾 (3)
            'tile': 3,
            'stone': 3,
            'brick': 3
        }
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            if 'labels' not in data:
                print(f"Warning: No 'labels' key in {json_file}")
                return False
                
            for label in data['labels']:
                try:
                    if 'name' not in label:
                        print(f"Warning: No 'name' field in label data in {json_file}")
                        continue
                        
                    class_name = label['name']
                    if class_name not in category_mapping:
                        print(f"Warning: Unknown class {class_name} in {json_file}")
                        continue
                        
                    # 直接使用大分类ID
                    category_id = category_mapping[class_name]
                    
                    required_keys = ['x1', 'y1', 'x2', 'y2']
                    if not all(key in label for key in required_keys):
                        print(f"Warning: Missing bbox coordinates in {json_file}")
                        continue
                    
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height)
                    
                    if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    else:
                        print(f"Warning: Invalid bbox values in {json_file}")
                        continue
                        
                except KeyError as e:
                    print(f"Warning: Missing key in label data in {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing label in {json_file}: {e}")
                    continue
                    
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return False
def create_augmentation_pipeline():
    """创建更温和的数据增强pipeline"""
    return A.Compose([
        # 亮度和对比度调整(更温和)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # 降低范围
            contrast_limit=0.1,    # 降低范围
            p=0.2
        ),
        # 色调和饱和度调整(更温和)
        A.HueSaturationValue(
            hue_shift_limit=10,    # 降低色调变化
            sat_shift_limit=15,    # 降低饱和度变化
            val_shift_limit=10,    # 降低明度变化
            p=0.2
        ),
        # CLAHE(更温和)
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.2
        ),
        # 水平翻转
        A.HorizontalFlip(p=0.5),
        # 轻微的位移、缩放和旋转
        A.ShiftScaleRotate(
            shift_limit=0.0625,    # 减小位移范围
            scale_limit=0.1,       # 减小缩放范围
            rotate_limit=5,        # 减小旋转角度
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),
        # 降噪和模糊(更温和)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 15.0), p=1),    # 降低噪声强度
            A.GaussianBlur(blur_limit=3, p=1),           # 降低模糊程度
            A.MedianBlur(blur_limit=3, p=1)              # 保持小的模糊核
        ], p=0.1)
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,    # 添加最小可见性阈值
        check_each_transform=True  # 检查每次转换是否有效
    ))



def load_yolo_bbox(txt_path):
    """加载YOLO格式的边界框"""
    bboxes = []
    class_labels = []
    
    if not os.path.exists(txt_path):
        return [], []
        
    with open(txt_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 5:
                class_label = int(data[0])
                x_center, y_center, width, height = map(float, data[1:])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_label)
    
    return bboxes, class_labels

def save_yolo_bbox(bboxes, class_labels, txt_path):
    """保存YOLO格式的边界框"""
    with open(txt_path, 'w') as f:
        for bbox, class_label in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_validation_set(num_augmentations=2):
    """对验证集进行温和的数据增强"""
    print("\nAugmenting validation set...")
    val_images_dir = os.path.join('val', 'images')
    val_labels_dir = os.path.join('val', 'labels')
    aug_images_dir = os.path.join('val_augmented', 'images')
    aug_labels_dir = os.path.join('val_augmented', 'labels')
    
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)
    
    # 复制原始文件
    for f in os.listdir(val_images_dir):
        shutil.copy2(
            os.path.join(val_images_dir, f),
            os.path.join(aug_images_dir, f)
        )
    for f in os.listdir(val_labels_dir):
        shutil.copy2(
            os.path.join(val_labels_dir, f),
            os.path.join(aug_labels_dir, f)
        )
    
    transform = create_augmentation_pipeline()
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(val_images_dir, img_file)
        label_path = os.path.join(val_labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = load_yolo_bbox(label_path)
        
        if not bboxes:
            continue
        
        for i in range(num_augmentations):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                aug_img_name = f"{os.path.splitext(img_file)[0]}_aug_{i+1}{os.path.splitext(img_file)[1]}"
                aug_label_name = f"{os.path.splitext(img_file)[0]}_aug_{i+1}.txt"
                
                aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(aug_images_dir, aug_img_name), aug_img)
                
                save_yolo_bbox(
                    augmented['bboxes'],
                    augmented['class_labels'],
                    os.path.join(aug_labels_dir, aug_label_name)
                )
                
            except Exception as e:
                print(f"Error in augmentation: {str(e)}")
                continue
    
    total_images = len([f for f in os.listdir(aug_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Augmented validation set contains {total_images} images")
def train_yolo(use_augmentation=False):
   """改进的YOLO训练配置，增加数据增强开关
   Args:
       use_augmentation (bool): 是否启用数据增强，默认为True
   """
   model = YOLO('yolo11n.pt')
   
   # 基础训练参数
   train_args = {
       'data': 'data.yaml',
       'epochs': 200,          # 可选: 300(更充分训练), 150(如果提前收敛)
       'imgsz': 640,          # 可选: 1024(更好的大目标检测), 512(更快的训练速度)
       'batch': 24,           # 可选: 32/48(GPU内存允许时), 16(内存不足时)
       'workers': 16,
       'device': '0',
       'patience': 50,        # 可选: 30(更快停止), 100(更有耐心)
       'save_period': 5,
       'exist_ok': True,
       'project': os.path.dirname(os.path.abspath(__file__)),
       'name': 'runs/train',
       
       # 优化器参数
       'optimizer': 'AdamW',  # 可选: 'SGD'(经典优化器), 'Adam'(不带权重衰减)
       'lr0': 0.0005,        # 可选: 0.001(更快收敛), 0.0001(更稳定)
       'lrf': 0.01,          # 可选: 0.05(更缓和衰减), 0.001(更激进衰减)
       'momentum': 0.937,     # 可选: 0.9(标准设置), 0.95(更强动量)
       'weight_decay': 0.0005,# 可选: 0.001(更强正则化), 0.0001(更弱正则化)
       'warmup_epochs': 10,   # 可选: 5(简单数据集), 15-20(复杂数据集)
       'warmup_momentum': 0.5,
       'warmup_bias_lr': 0.05,
       
       # 损失函数权重
       'box': 4.0,           # 可选: 5.0(如果定位不准)
       'cls': 2.0,           # 可选: 1.5-2.0(如果分类不准)
       'dfl': 1.5,           # 可选: 2.0(如果需要更精确的定位)
       
       # 其他基础配置
       'close_mosaic': 0,
       'nbs': 64,            # 可选: 32或128
       'overlap_mask': False,
       'multi_scale': False,
       'single_cls': False,
   }
   
   # 数据集较大时的优化配置示例:
   # train_args.update({
   #     'batch': 32,
   #     'lr0': 0.001,
   #     'epochs': 150,
   #     'patience': 30
   # })
   
   # 数据集较小时的优化配置示例:
   # train_args.update({
   #     'batch': 16,
   #     'lr0': 0.0001,
   #     'weight_decay': 0.001,
   #     'warmup_epochs': 15
   # })
   
   # 注重检测精度时的优化配置示例:
   # train_args.update({
   #     'imgsz': 1024,
   #     'box': 5.0,
   #     'dfl': 2.0,
   #     'patience': 100
   # })
   
   # 注重训练速度时的优化配置示例:
   # train_args.update({
   #     'imgsz': 512,
   #     'epochs': 150,
   #     'patience': 30,
   #     'batch': 48  # 如果GPU内存允许
   # })
   
   # 数据增强参数，仅在use_augmentation=True时添加
   if use_augmentation:
       augmentation_args = {
           'augment': True,
           'degrees': 5.0,
           'scale': 0.2,
           'fliplr': 0.5,
           'flipud': 0.0,
           'hsv_h': 0.01,
           'hsv_s': 0.2,
           'hsv_v': 0.1,
           'mosaic': 0,  # 保持关闭复杂增强
           'mixup': 0,
           'copy_paste': 0,
       }
       train_args.update(augmentation_args)
   else:
       # 关闭所有数据增强
       train_args.update({
           'augment': False,
           'degrees': 0.0,
           'scale': 0.0,
           'fliplr': 0.0,
           'flipud': 0.0,
           'hsv_h': 0.0,
           'hsv_s': 0.0,
           'hsv_v': 0.0,
           'mosaic': 0,
           'mixup': 0,
           'copy_paste': 0,
       })
   
   results = model.train(**train_args)


def main():
    try:
        # 设置数据目录
        data_dir = datapath
        
        # 1. 检查数据集
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(data_dir)
        
        # 2. 创建配置文件
        print("\nStep 2: Creating data.yaml...")
        create_data_yaml()
        
        # 3. 准备数据集 (不再增强验证集)
        print("\nStep 3: Preparing dataset...")
        train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs)
        
        if val_size < 5:  # 降低最小验证集大小要求
            raise ValueError(f"Validation set too small ({val_size} images). Need at least 5 images.")
            
        # 4. 开始训练
        print("\nStep 4: Starting training...")
        train_yolo()
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
