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
    print("Checking dataset integrity...")
    
    # 获取所有图片和标签文件
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_pairs = []
    
    print(f"Found {len(image_files)} total images")
    
    # 检查每个图片是否有效且有对应的标签文件
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        json_file = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.json')
        
        # 检查图片完整性
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Corrupted or invalid image file: {img_file}")
                continue
                
            # 检查图片尺寸是否合理
            height, width = img.shape[:2]
            if height < 10 or width < 10:
                print(f"Warning: Image too small: {img_file}")
                continue
                
        except Exception as e:
            print(f"Warning: Error reading image {img_file}: {e}")
            continue
            
        # 检查标签文件
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    
                # 验证标签数据结构
                if 'labels' not in label_data:
                    print(f"Warning: Invalid label structure in {json_file}")
                    continue
                    
                valid_pairs.append(img_file)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON file: {json_file}")
                continue
        else:
            print(f"Warning: No label file for {img_file}")
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    return valid_pairs

def create_data_yaml():
    """创建数据配置文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        'path': current_dir,
        'train': os.path.join(current_dir, 'train/images'), 
        'val': os.path.join(current_dir, 'val/images'),  # 修改为正确的验证集路径
        'test': os.path.join(current_dir, 'test/images'),
        'names': {
            0: 'potato', 1: 'daikon', 2: 'carrot',
            3: 'bottle', 4: 'can', 5: 'battery',
            6: 'drug', 7: 'inner_packing',
            8: 'tile', 9: 'stone', 10: 'brick'
        },
        'nc': 11
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
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
            src_img = os.path.join(data_dir, img_file)
            src_json = os.path.join(data_dir, img_file.replace('.jpg', '.json'))
            dst_img = os.path.join(split_name, 'images', img_file)
            dst_txt = os.path.join(split_name, 'labels', img_file.replace('.jpg', '.txt'))
            
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
    """转换标签文件从新的JSON格式到YOLO格式"""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False
            
        # 获取图片路径
        img_path = json_file.replace('.json', '.jpg')
        if not os.path.exists(img_path):
            img_path = json_file.replace('.json', '.png')
        
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image: {img_path}")
            return False
        
        img_height, img_width = img.shape[:2]
        
        # 读取JSON标签
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 类别映射
        class_mapping = {
            'potato': 0, 'daikon': 1, 'carrot': 2,
            'bottle': 3, 'can': 4, 'battery': 5,
            'drug': 6, 'inner_packing': 7,
            'tile': 8, 'stone': 9, 'brick': 10
        }
        
        # 写入YOLO格式标签
        with open(txt_file, 'w', encoding='utf-8') as f:
            for label in data['labels']:
                try:
                    class_name = label['name']
                    if class_name not in class_mapping:
                        print(f"Warning: Unknown class {class_name} in {json_file}")
                        continue
                        
                    class_id = class_mapping[class_name]
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except KeyError as e:
                    print(f"Warning: Missing key in label data in {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing label in {json_file}: {e}")
                    continue
        return True
        
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
def train_yolo():
    """改进的YOLO训练配置"""
    model = YOLO('yolo11n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=200,  # 增加到200轮
        imgsz=640,
        batch=16,
        workers=8,
        device='0',
        patience=50,
        save_period=5,
        exist_ok=True,
        project=os.path.dirname(os.path.abspath(__file__)),
        name='runs/train',
        
        # 优化器参数调整
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.01,
        # scheduler='cosine',  # 移除无效参数
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,
        warmup_momentum=0.5,
        warmup_bias_lr=0.05,
        
        # 损失函数权重调整
        box=4.0,
        cls=1.0,
        dfl=1.5,
        
        # 基础数据增强参数
        augment=True,
        degrees=5.0,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.1,
        
        # 关闭复杂的数据增强
        mosaic=0,
        mixup=0,
        copy_paste=0,
        
        # 添加早停和模型评估配置
        close_mosaic=0,
        nbs=64,
        overlap_mask=False,
        multi_scale=False,
        single_cls=False,
        
        # 启用混合精度
        # precision=16,  # 移除无效参数
    )


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
