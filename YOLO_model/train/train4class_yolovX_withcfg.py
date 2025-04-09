"""add path of ttf font"""

import os
import shutil
from pathlib import Path


def setup_font():
    config_dir = Path.home() / ".config" / "Ultralytics"
    target_font = config_dir / "Arial.Unicode.ttf"
    if target_font.exists():
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    local_font = Path(__file__).parent / "Arial.Unicode.ttf"
    if not local_font.exists():
        print(
            f"Please place the Arial.Unicode.ttf file in the code directory: {local_font}"
        )
        return
    try:
        shutil.copy2(local_font, target_font)
        print(f"Font file has been copied to: {target_font}")
    except Exception as e:
        print(f"Error occurred while copying the font file: {e}")


setup_font()
"""ttf added """
from ultralytics import YOLO
import yaml
import json
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np
from pathlib import Path
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
import yaml

class ConfigManager:
    """Manages training configurations, loading/saving from a YAML file."""
    def __init__(self, config_file="traincfg.yaml"):
        self.config_file = Path(config_file)
        # Default configuration values
        self.config = {
            "select_model": "yolo11n.pt",
            "datapath": "./label",
            "use_augmentation": False,
            "use_mixed_precision": True, # Defaulting to True as in the original main call
            "train_config_profile": "focus_accuracy", # Defaulting as in the original main call
            "resume_training": False,
            # Add other potentially configurable parameters from train_yolo here
            "epochs": 120,
            "imgsz": 640,
            "batch_gpu": 10, # Default GPU batch size
            "batch_cpu": 4,  # Default CPU batch size
            "workers_gpu": 8, # Max workers for GPU
            "workers_cpu": 4, # Max workers for CPU
            "patience": 15,
            "save_period": 5,
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 10,
            "warmup_momentum": 0.5,
            "warmup_bias_lr": 0.05,
            "box_loss_gain": 4.0,
            "cls_loss_gain": 2.0,
            "dfl_loss_gain": 1.5,
            "close_mosaic_epochs": 0,
            "nominal_batch_size": 64,
            "overlap_mask": False,
            "multi_scale": True,
            "single_cls": False,
            "rect_training": True,
            "cache_images": True,
            # Augmentation specific defaults (can be overridden if use_augmentation is True)
            "degrees": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
        self._load_config() # Load config on initialization

    def _load_config(self):
        """Loads configuration from the YAML file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config: # Check if file is not empty
                         # Update defaults only with keys present in the file
                        for key, value in loaded_config.items():
                            if key in self.config:
                                self.config[key] = value
                            else:
                                print(f"Warning: Ignoring unknown key '{key}' in {self.config_file}")
                print(f"Configuration loaded from {self.config_file}")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {self.config_file}: {e}. Using default config.")
            except Exception as e:
                print(f"Error loading config file {self.config_file}: {e}. Using default config.")
        else:
            print(f"Configuration file {self.config_file} not found. Using default config and creating the file.")
            self.save_config() # Create the file with defaults if it doesn't exist

    def save_config(self):
        """Saves the current configuration to the YAML file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file {self.config_file}: {e}")

    def get(self, key, default=None):
        """Gets a configuration value by key."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Sets a configuration value by key."""
        if key in self.config:
            self.config[key] = value
        else:
            # Optionally allow adding new keys or warn/error
            # self.config[key] = value # Allow adding new keys
             print(f"Warning: Key '{key}' not found in the default configuration keys. It will be added.")
             self.config[key] = value # Or raise an error: raise KeyError(f"Key '{key}' not found.")

    def get_all(self):
        """Returns the entire configuration dictionary."""
        return self.config.copy()

def validate_json_file(json_path):
    """
    验证 JSON 文件是否有效。
    返回 True 表示有效，False 表示无效。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
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

# Helper function to check a single image and its label file
def _check_single_file(img_file, data_dir):
    """Checks a single image file and its corresponding JSON label file."""
    img_path = os.path.join(data_dir, img_file)
    base_name = os.path.splitext(img_file)[0]
    json_file = os.path.join(data_dir, base_name + ".json")

    # 检查图片完整性
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 损坏或无效的图片文件: {img_file}")
            return None # Indicate failure
        # 检查图片尺寸是否合理
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            print(f"警告: 图片尺寸过小: {img_file}")
            return None # Indicate failure
    except Exception as e:
        print(f"警告: 读取图片 {img_file} 时出错: {e}")
        return None # Indicate failure

    # 检查标签文件是否存在且有效
    if os.path.exists(json_file):
        # 确保标签文件确实是一个 JSON 文件
        if not json_file.lower().endswith(".json"):
            print(f"警告: 标签文件扩展名不正确 (应为 .json): {json_file}")
            return None # Indicate failure

        # 验证 JSON 文件内容 (first check validity)
        if not validate_json_file(json_file):
             # validate_json_file already printed a warning
            return None # Indicate failure

        # 验证 JSON 文件结构 (second check structure)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            if "labels" not in label_data:
                print(f"警告: 标签文件结构无效 (缺少 'labels' 键): {json_file}")
                return None # Indicate failure
            # If all checks pass, return the image file name
            return img_file
        except Exception as e:
            # Catch errors during the structure check read
            print(f"警告: 处理标签文件 {json_file} (结构检查) 时发生错误: {e}")
            return None # Indicate failure
    else:
        print(f"警告: 找不到对应的标签文件: {json_file}")
        return None # Indicate failure


def check_and_clean_dataset(data_dir):
    """检查数据集完整性并清理无效数据 (Parallelized Version)"""
    print("正在检查数据集完整性 (并行)...")
    image_extensions = (".jpg", ".jpeg", ".png")

    try:
        # Check if data_dir exists before listing
        if not os.path.isdir(data_dir):
             print(f"错误: 数据目录不存在或不是一个目录: {data_dir}")
             return []
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"错误: 无法列出目录 {data_dir}: {e}")
        return []

    image_files = [
        f for f in all_files if f.lower().endswith(image_extensions)
    ]
    print(f"找到 {len(image_files)} 个潜在图片文件")
    if not image_files:
        print("目录中未找到支持的图片文件。")
        return []

    valid_pairs = []
    futures = []
    # Determine max_workers, leave some cores free for other tasks
    # Use at least 1 worker even if cpu_count is None or 1
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"使用最多 {max_workers} 个 worker 线程进行检查...")

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit check task for each image file
        for img_file in image_files:
            futures.append(executor.submit(_check_single_file, img_file, data_dir))

        # Collect results as they complete
        for future in futures:
            try:
                result = future.result() # Get the return value from _check_single_file
                if result:  # If the helper function returned an img_file name (success)
                    valid_pairs.append(result)
            except Exception as exc:
                # Catch potential exceptions during future execution/result retrieval
                # Although _check_single_file handles most internal errors
                print(f'一个文件检查任务产生异常: {exc}')

    print(f"\n检查完成。找到 {len(valid_pairs)} 对有效的图片和标签文件。")
    return valid_pairs

def create_data_yaml():
    """创建数据配置文件 - 使用四大分类"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        "path": current_dir,
        "train": os.path.join(current_dir, "train/images"),
        "val": os.path.join(current_dir, "val/images"),
        "test": os.path.join(current_dir, "test/images"),
        "names": {0: "厨余垃圾", 1: "可回收垃圾", 2: "有害垃圾", 3: "其他垃圾"},
        "nc": 4,  # 改为4个大类
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def process_split(split_name, files, data_dir):
    """Processes image copying and label conversion for a given dataset split."""
    print(f"\nProcessing {split_name} split...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    # Ensure target directories exist (though prepare_dataset already creates them)
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file) # Corrected destination path
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt") # Corrected destination path

        # Copy image and convert label
        # Error handling during conversion is inside convert_labels
        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                shutil.copy2(src_img, dst_img)
                convert_labels(src_json, dst_txt)
            except Exception as e:
                print(f"Error processing file pair ({img_file}, {base_name}.json): {e}")
        else:
            if not os.path.exists(src_img):
                 print(f"Warning: Source image not found during split processing: {src_img}")
            if not os.path.exists(src_json):
                 print(f"Warning: Source JSON not found during split processing: {src_json}")


    print(f"{split_name}: Processed {len(files)} potential images")


def prepare_dataset(data_dir, valid_pairs):
    """准备数据集 - 修改验证集划分比例 (增强版，处理非空目录)"""
    # 确保验证集至少有10张图片
    if len(valid_pairs) < 15:
        raise ValueError(
            f"有效数据对数量不足 ({len(valid_pairs)})。至少需要15张图片。"
        )
    
    # 清理现有目录 - 改进的清理方法
    print("\n正在清理现有的train/val/test目录...")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), split)
        if os.path.exists(split_path):
            print(f"正在删除 {split_path} 目录...")
            try:
                # 尝试直接删除目录
                shutil.rmtree(split_path)
            except OSError as e:
                print(f"无法直接删除目录 {split_path}，错误：{e}")
                print(f"尝试逐个删除其中的文件和子目录...")
                
                # 逐个删除文件和目录
                for root, dirs, files in os.walk(split_path, topdown=False):
                    # 先删除文件
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"已删除文件: {file_path}")
                        except Exception as e:
                            print(f"警告: 无法删除文件 {file_path}: {e}")
                    
                    # 再删除空目录
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                            print(f"已删除目录: {dir_path}")
                        except Exception as e:
                            print(f"警告: 无法删除目录 {dir_path}: {e}")
                
                # 最后尝试删除主目录
                try:
                    os.rmdir(split_path)
                    print(f"已成功删除 {split_path}")
                except Exception as e:
                    print(f"警告: 无法删除主目录 {split_path}: {e}")
                    print(f"将尝试继续处理...")
        
        # 重新创建目录结构
        print(f"创建 {split} 目录结构...")
        for subdir in ["images", "labels"]:
            subdir_path = os.path.join(split_path, subdir)
            try:
                os.makedirs(subdir_path, exist_ok=True)
                print(f"已创建目录: {subdir_path}")
            except Exception as e:
                print(f"错误: 无法创建目录 {subdir_path}: {e}")
                raise

    # 数据集划分 (80% 训练, 10% 验证, 10% 测试)
    print("将数据集划分为训练集、验证集和测试集...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # 使用 ThreadPoolExecutor 并行处理每个数据集分割
    print("开始并行处理数据集划分...")
    # 确定worker数量，为其他任务保留一些核心
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个划分提交任务
        futures = {
            executor.submit(process_split, split_name, files, data_dir): split_name
            for split_name, files in splits.items()
        }

        # 等待所有任务完成并检查异常
        for future in futures:
            split_name = futures[future]
            try:
                future.result()  # 等待任务完成并引发异常（如果有）
                print(f"完成处理 {split_name} 划分。")
            except Exception as exc:
                print(f'{split_name} 划分生成了一个异常: {exc}')
                print(f'尝试继续处理其他划分...')

    print("\n数据集准备完成。")
    print(f"训练集: {len(train_files)} 图片, 验证集: {len(val_files)} 图片, 测试集: {len(test_files)} 图片")
    return len(train_files), len(val_files), len(test_files)


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """转换边界框从x1,y1,x2,y2到YOLO格式"""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

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
    """转换为四大类"""
    try:
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found: {json_file}")
            return False
        # 获取图片路径

        base_name = os.path.splitext(json_file)[0]
        possible_extensions = [".jpg", ".jpeg", ".png"]
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

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 细分类到大分类的映射

        category_mapping = {
            # 厨余垃圾 (0)
            "potato": 0,
            "daikon": 0,
            "carrot": 0,
            # 可回收垃圾 (1)
            "bottle": 1,
            "can": 1,
            # 有害垃圾 (2)
            "battery": 2,
            "drug": 2,
            "inner_packing": 2,
            # 其他垃圾 (3)
            "tile": 3,
            "stone": 3,
            "brick": 3,
        }

        with open(txt_file, "w", encoding="utf-8") as f:
            if "labels" not in data:
                print(f"Warning: No 'labels' key in {json_file}")
                return False
            for label in data["labels"]:
                try:
                    if "name" not in label:
                        print(f"Warning: No 'name' field in label data in {json_file}")
                        continue
                    class_name = label["name"]
                    if class_name not in category_mapping:
                        print(f"Warning: Unknown class {class_name} in {json_file}")
                        continue
                    # 直接使用大分类ID

                    category_id = category_mapping[class_name]

                    required_keys = ["x1", "y1", "x2", "y2"]
                    if not all(key in label for key in required_keys):
                        print(f"Warning: Missing bbox coordinates in {json_file}")
                        continue
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height
                    )

                    if all(
                        0 <= val <= 1 for val in [x_center, y_center, width, height]
                    ):
                        f.write(
                            f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )
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


def load_yolo_bbox(txt_path):
    """加载YOLO格式的边界框"""
    bboxes = []
    class_labels = []

    if not os.path.exists(txt_path):
        return [], []
    with open(txt_path, "r") as f:
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
    with open(txt_path, "w") as f:
        for bbox, class_label in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(
                f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


def train_yolo(cfg_manager: ConfigManager):
    """
    YOLO训练配置, 使用 ConfigManager 获取所有参数。
    """
    # Get parameters from ConfigManager
    select_model = cfg_manager.get("select_model")
    use_augmentation = cfg_manager.get("use_augmentation")
    use_mixed_precision = cfg_manager.get("use_mixed_precision")
    config_profile = cfg_manager.get("train_config_profile")
    resume = cfg_manager.get("resume_training")

    model = YOLO(select_model)  # 加载预训练的YOLO模型权重

    device = "cpu"
    if torch.cuda.is_available():
        device = "0" # Default to the first GPU

    # Determine device-specific settings
    if device == "cpu":
        batch_size = cfg_manager.get("batch_cpu")
        workers = max(1, min(os.cpu_count() - 1, cfg_manager.get("workers_cpu")))
        actual_mixed_precision = False # CPU不支持混合精度训练
    else: # GPU
        batch_size = cfg_manager.get("batch_gpu")
        # Calculate workers based on available CPUs, capped by config and a practical limit (e.g., 8 or 16)
        max_workers_cfg = cfg_manager.get("workers_gpu")
        cpu_cores = os.cpu_count() if os.cpu_count() else 1
        workers = max(1, min(cpu_cores - 2 if cpu_cores > 2 else 1, max_workers_cfg))
        actual_mixed_precision = use_mixed_precision # Use config setting for GPU

    # Base training arguments from ConfigManager
    train_args = {
        "data": "data.yaml", # Assumes data.yaml is always generated in the script dir
        "epochs": cfg_manager.get("epochs"),
        "imgsz": cfg_manager.get("imgsz"),
        "batch": batch_size,
        "workers": workers,
        "device": device,
        "patience": cfg_manager.get("patience"),
        "save_period": cfg_manager.get("save_period"),
        "exist_ok": True, # Allow overwriting previous runs in the same folder
        "project": os.path.dirname(os.path.abspath(__file__)), # Project dir is script dir
        "name": "runs/train", # Default run name
        "optimizer": cfg_manager.get("optimizer"),
        "lr0": cfg_manager.get("lr0"),
        "lrf": cfg_manager.get("lrf"),
        "momentum": cfg_manager.get("momentum"),
        "weight_decay": cfg_manager.get("weight_decay"),
        "warmup_epochs": cfg_manager.get("warmup_epochs"),
        "warmup_momentum": cfg_manager.get("warmup_momentum"),
        "warmup_bias_lr": cfg_manager.get("warmup_bias_lr"),
        "box": cfg_manager.get("box_loss_gain"),
        "cls": cfg_manager.get("cls_loss_gain"),
        "dfl": cfg_manager.get("dfl_loss_gain"),
        "close_mosaic": cfg_manager.get("close_mosaic_epochs"), # Uses epoch count directly
        "nbs": cfg_manager.get("nominal_batch_size"),
        "overlap_mask": cfg_manager.get("overlap_mask"),
        "multi_scale": cfg_manager.get("multi_scale"),
        "single_cls": cfg_manager.get("single_cls"),
        "rect": cfg_manager.get("rect_training"),
        "cache": cfg_manager.get("cache_images"),
        "resume": resume,
        "half": actual_mixed_precision # Set based on device capability and config
    }

    # Apply profile-specific overrides
    if config_profile == "large_dataset":
        train_args.update({
            "batch": cfg_manager.get("batch_gpu") * 2 if device == "0" else cfg_manager.get("batch_cpu"), # Example override
            "lr0": 0.001,
            "epochs": 150,
            "patience": 30,
        })
    elif config_profile == "small_dataset":
        train_args.update({
             "batch": cfg_manager.get("batch_gpu") // 2 if device == "0" and cfg_manager.get("batch_gpu") > 1 else cfg_manager.get("batch_cpu"), # Example override
             "lr0": 0.0001,
             "weight_decay": 0.001,
             "warmup_epochs": 15,
        })
    elif config_profile == "focus_accuracy":
         train_args.update({
             "imgsz": 640, # Ensure high res
             "box": 6.0,
             "cls": 3.0,
             "dfl": 2.5,
             "patience": 100,
             "batch": cfg_manager.get("batch_gpu") if device == "0" else cfg_manager.get("batch_cpu"), # Standard batch for accuracy focus
         })
    elif config_profile == "focus_speed":
        train_args.update({
            "imgsz": 512, # Smaller image size
            "epochs": 150, # More epochs might still be needed
            "patience": 30,
            "batch": cfg_manager.get("batch_gpu") * 3 if device == "0" else cfg_manager.get("batch_cpu"), # Larger batch for speed if GPU allows
            "half": True if device == "0" else False, # Force mixed precision on GPU for speed
        })
    elif config_profile != "default":
        print(f"警告: 未识别的训练配置模式 '{config_profile}'，将使用基础配置。")


    # Apply augmentation settings if enabled
    if use_augmentation:
        augmentation_args = {
            "augment": True,
            "degrees": cfg_manager.get("degrees", 5.0), # Provide defaults if not in base config
            "scale": cfg_manager.get("scale", 0.2),
            "fliplr": cfg_manager.get("fliplr", 0.5),
            "flipud": cfg_manager.get("flipud", 0.0),
            "hsv_h": cfg_manager.get("hsv_h", 0.01),
            "hsv_s": cfg_manager.get("hsv_s", 0.2),
            "hsv_v": cfg_manager.get("hsv_v", 0.1),
            "mosaic": cfg_manager.get("mosaic", 0.0), # Default mosaic off unless specified
            "mixup": cfg_manager.get("mixup", 0.0),
            "copy_paste": cfg_manager.get("copy_paste", 0.0),
        }
        train_args.update(augmentation_args)
    else:
        # Ensure augmentations are explicitly off if use_augmentation is False
        no_augment_args = {
            "augment": False,
            "degrees": 0.0, "scale": 0.0, "fliplr": 0.0, "flipud": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        }
        train_args.update(no_augment_args)

    # Start training
    try:
        print(f"\n--- Training Configuration ---")
        print(f"Using model: {select_model}")
        print(f"Device: {'GPU 0' if device == '0' else 'CPU'}")
        print(f"Profile: {config_profile}")
        print(f"Resume: {resume}")
        print(f"Augmentation: {'Enabled' if train_args['augment'] else 'Disabled'}")
        print(f"Mixed Precision: {'Enabled' if train_args['half'] else 'Disabled'}")
        print(f"Batch Size: {train_args['batch']}")
        print(f"Workers: {train_args['workers']}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Image Size: {train_args['imgsz']}")
        print(f"----------------------------\n")

        results = model.train(**train_args)
        return results
    except Exception as e:
        print(f"Training error: {str(e)}")
        # Consider re-raising or specific handling
        # raise # Re-raise the exception
        return None


def main():
    try:
        # Initialize ConfigManager - this will load or create traincfg.yaml
        cfg_manager = ConfigManager()

        # Optional: Allow setting specific config via code before saving/using
        # cfg_manager.set("epochs", 150)
        # cfg_manager.save_config() # Save if changed

        # Clear CUDA cache and collect garbage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get datapath from config
        data_dir = cfg_manager.get("datapath")
        if not data_dir or not Path(data_dir).is_dir():
             print(f"Error: Invalid 'datapath' in configuration: {data_dir}")
             # Create a default directory or raise error?
             default_data_dir = Path("./label_default")
             default_data_dir.mkdir(exist_ok=True)
             print(f"Warning: 'datapath' invalid. Using default: {default_data_dir}")
             cfg_manager.set("datapath", str(default_data_dir))
             cfg_manager.save_config() # Save the corrected path
             data_dir = str(default_data_dir)
             # Or: raise ValueError(f"Invalid 'datapath' in configuration: {data_dir}")


        # 1. Check dataset
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(data_dir)
        gc.collect()

        # 2. Create data.yaml (uses current directory, independent of config for now)
        print("\nStep 2: Creating data.yaml...")
        create_data_yaml() # This function defines paths relative to the script

        # 3. Prepare dataset splits
        print("\nStep 3: Preparing dataset...")
        train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs)
        gc.collect()
        if val_size < 5:
            # Optionally adjust config if val set is too small, or just error out
            print(f"Warning: Validation set small ({val_size} images). Training might be unstable.")
            # raise ValueError(f"Validation set too small ({val_size} images). Need at least 5.")

        # 4. Start training using ConfigManager
        print("\nStep 4: Starting training...")
        train_yolo(cfg_manager) # Pass the manager instance

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        # Consider logging the full traceback for debugging
        import traceback
        traceback.print_exc()
        # raise # Re-raise if you want the script to exit with an error code

if __name__ == "__main__":
    main()

