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

def check_and_clean_dataset():
    """检查数据集完整性并清理无效数据"""
    print("Checking dataset integrity...")
    
    # 检查必要的目录
    required_dirs = ['picture', 'label']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Required directory '{dir_name}' not found")
    
    # 获取所有图片和标签文件
    image_files = [f for f in os.listdir('picture') if f.endswith(('.jpg', '.jpeg', '.png'))]
    valid_pairs = []
    
    print(f"Found {len(image_files)} total images")
    
    # 检查每个图片是否有对应的标签文件
    for img_file in image_files:
        img_path = os.path.join('picture', img_file)
        json_file = os.path.join('label', img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
        
        if os.path.exists(json_file):
            try:
                # 验证JSON文件是否有效
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
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
        'val': os.path.join(current_dir, 'val_augmented/images'),
        'test': os.path.join(current_dir, 'test/images'),
        'names': {
            0: 'battery', 1: 'expired_drug', 2: 'inner_packing',
            3: 'can', 4: 'bottle', 5: 'potato', 6: 'daikon',
            7: 'carrot', 8: 'tile', 9: 'pebble', 10: 'brick'
        },
        'nc': 11
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def prepare_dataset(valid_pairs):
    """准备数据集"""
    # 确保验证集至少有3张图片
    if len(valid_pairs) < 10:
        raise ValueError(f"Not enough valid data pairs ({len(valid_pairs)}). Need at least 10 images.")
    
    # 清理现有目录
    for split in ['train', 'val', 'test', 'val_augmented']:
        if os.path.exists(split):
            shutil.rmtree(split)
        if split != 'val_augmented':  # val_augmented将在后面创建
            for subdir in ['images', 'labels']:
                os.makedirs(os.path.join(split, subdir), exist_ok=True)
    
    # 数据集划分 (60% 训练, 20% 验证, 20% 测试)
    train_files, temp = train_test_split(valid_pairs, test_size=0.4, random_state=42)
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
            src_img = os.path.join('picture', img_file)
            src_json = os.path.join('label', img_file.replace('.jpg', '.json'))
            dst_img = os.path.join(split_name, 'images', img_file)
            dst_txt = os.path.join(split_name, 'labels', img_file.replace('.jpg', '.txt'))
            
            # 复制图片和转换标签
            shutil.copy2(src_img, dst_img)
            convert_labels(src_json, dst_txt)
            
        print(f"{split_name}: {len(files)} images")
    
    return len(train_files), len(val_files), len(test_files)

def create_augmentation_pipeline():
    """创建温和的数据增强pipeline"""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        A.CLAHE(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.3
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def convert_labelme_to_yolo(points, img_width, img_height):
    """转换标注格式从LabelMe到YOLO"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    x_center = (xmin + xmax) / (2 * img_width)
    y_center = (ymin + ymax) / (2 * img_height)
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def convert_labels(json_file, txt_file):
    """转换标签文件从JSON到YOLO格式"""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_width = float(data['imageWidth'])
        img_height = float(data['imageHeight'])
        
        if not os.path.exists('label/classes.txt'):
            print(f"Warning: classes.txt not found in label directory")
            return False
            
        with open('label/classes.txt', 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for shape in data['shapes']:
                try:
                    class_id = classes.index(shape['label'])
                    points = shape['points']
                    
                    x_center, y_center, width, height = convert_labelme_to_yolo(
                        points, img_width, img_height)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except ValueError as e:
                    print(f"Warning: Invalid class label in {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing shape in {json_file}: {e}")
                    continue
        return True
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return False

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
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=20,
        workers=8,
        device='0',
        patience=20,
        save_period=10,
        exist_ok=True,
        project=os.path.dirname(os.path.abspath(__file__)),
        name='runs/train',
        
        # 关闭 Mosaic 和相关的数据增强
        mosaic=0,      # 关闭 Mosaic
        mixup=0,       # 关闭 Mixup
        copy_paste=0,  # 关闭 Copy-Paste
        
        # 保留基础的数据增强
        augment=True,  # 保留基础增强
        degrees=10.0,  # 减小旋转角度
        scale=0.3,     # 缩放范围
        fliplr=0.5,    # 水平翻转
        flipud=0.0,    # 不使用垂直翻转
        hsv_h=0.015,   # HSV色调
        hsv_s=0.4,     # HSV饱和度
        hsv_v=0.3,     # HSV亮度
        
        # 优化器参数
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.1,
        momentum=0.9,
        weight_decay=0.001,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 损失函数权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

def main():
    try:
        # 1. 检查数据集
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset()
        
        # 2. 创建配置文件
        print("\nStep 2: Creating data.yaml...")
        create_data_yaml()
        
        # 3. 准备数据集
        print("\nStep 3: Preparing dataset...")
        train_size, val_size, test_size = prepare_dataset(valid_pairs)
        
        if val_size < 2:
            raise ValueError(f"Validation set too small ({val_size} images). Need at least 2 images.")
            
        # 4. 增强验证集
        print("\nStep 4: Augmenting validation set...")
        augment_validation_set(num_augmentations=2)
        
        # 5. 开始训练
        print("\nStep 5: Starting training...")
        train_yolo()
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()