import os
import shutil
from pathlib import Path
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

select_model = "yolo12s.pt"  # 选择的模型,默认为yolo12s,可以更改
datapath = "./label"  # 根据实际情况修改

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

def try_create_symlink(src, dst):
    """Try to create a symbolic link and return True if successful, False otherwise."""
    try:
        # Create the symbolic link
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception as e:
        print(f"Warning: Could not create symbolic link from {src} to {dst}: {e}")
        return False


def process_split(split_name, files, data_dir, use_symlinks=True):
    """
    Processes image copying/linking and label conversion for a given dataset split.
    
    Args:
        split_name: Name of the split (train, val, test)
        files: List of image filenames to process
        data_dir: Source directory containing the original files
        use_symlinks: Whether to try using symbolic links instead of copying (default: True)
    """
    print(f"\nProcessing {split_name} split...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    # Ensure target directories exist
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)
    
    # First file - try to create a symlink to test if it works
    symlinks_working = False
    if use_symlinks and files:
        test_img = os.path.join(data_dir, files[0])
        test_dst = os.path.join(split_img_dir, "test_symlink_" + files[0])
        symlinks_working = try_create_symlink(test_img, test_dst)
        
        # Clean up the test link
        if os.path.exists(test_dst):
            os.remove(test_dst)
    
    print(f"{split_name}: Using {'symbolic links' if symlinks_working else 'file copying'} for dataset preparation")

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file)
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt")

        # Process the image and label files
        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                # Handle the image file - try symlink first if enabled
                if symlinks_working:
                    if not try_create_symlink(src_img, dst_img):
                        # If symlink creation fails, fall back to copying
                        shutil.copy2(src_img, dst_img)
                else:
                    # Use direct copy if symlinks aren't working
                    shutil.copy2(src_img, dst_img)
                
                # Convert the label (no need to change this part)
                convert_labels(src_json, dst_txt)
            except Exception as e:
                print(f"Error processing file pair ({img_file}, {base_name}.json): {e}")
        else:
            if not os.path.exists(src_img):
                print(f"Warning: Source image not found during split processing: {src_img}")
            if not os.path.exists(src_json):
                print(f"Warning: Source JSON not found during split processing: {src_json}")

    print(f"{split_name}: Processed {len(files)} potential images")


def prepare_dataset(data_dir, valid_pairs, use_symlinks=True):
    """准备数据集 - 支持使用符号链接而不是复制文件来节省空间
    
    Args:
        data_dir: 包含图像和标签文件的源目录
        valid_pairs: 有效的图像文件名列表
        use_symlinks: 是否尝试使用符号链接而不是复制文件 (默认: True)
    """
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

    # 数据集划分 (90% 训练, 5% 验证, 5% 测试)
    print("将数据集划分为训练集、验证集和测试集...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.1, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # 使用 ThreadPoolExecutor 并行处理每个数据集分割
    print("开始并行处理数据集划分...")
    # 确定worker数量，为其他任务保留一些核心
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个划分提交任务，包括use_symlinks参数
        futures = {
            executor.submit(process_split, split_name, files, data_dir, use_symlinks): split_name
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
            "Kitchen_waste": 0,
            "potato": 0,
            "daikon": 0,
            "carrot": 0,
            # 可回收垃圾 (1)
            "Recyclable_waste": 1,
            "bottle": 1,
            "can": 1,
            # 有害垃圾 (2)
            "Hazardous_waste": 2,
            "battery": 2,
            "drug": 2,
            "inner_packing": 2,
            # 其他垃圾 (3)
            "Other_waste": 3,
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

def save_quantized_models(weights_dir, data_yaml_path):
    """加载最佳模型并保存FP16版本"""
    import shutil
    
    best_pt_path = os.path.join(weights_dir, 'best.pt')
    if not os.path.exists(best_pt_path):
        print(f"错误: {best_pt_path} 未找到，无法保存量化模型。")
        return

    print(f"\n正在从 {best_pt_path} 加载最佳模型...")
    try:
        model = YOLO(best_pt_path)
    except Exception as e:
        print(f"加载模型 {best_pt_path} 时出错: {e}")
        return

    # 已移除FP32权重保存部分

    # 创建FP16模型
    print("\n创建FP16模型...")
    # 复制原始模型
    fp16_model_path = os.path.join(weights_dir, 'best_fp16.pt')
    shutil.copy(best_pt_path, fp16_model_path)
    
    # 加载副本并转换为FP16
    fp16_model = YOLO(fp16_model_path)
    if hasattr(fp16_model, 'model'):
        fp16_model.model = fp16_model.model.half()
        # 保存转换后的模型
        fp16_model.save(fp16_model_path)
        print(f"FP16模型已保存至 {fp16_model_path}")
    else:
        print("无法转换为FP16模型：模型结构不符合预期")
    
    # 清理内存
    del fp16_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 导出TorchScript模型（可选但有用）
    try:
        print("\n尝试导出TorchScript格式模型...")
        torchscript_results = model.export(format='torchscript')
        if hasattr(torchscript_results, 'saved_model'):
            ts_path = torchscript_results.saved_model
            print(f"TorchScript模型已导出至: {ts_path}")
            
            # 移动到目标位置（如果导出位置不在weights_dir中）
            if os.path.dirname(ts_path) != weights_dir:
                ts_target_path = os.path.join(weights_dir, 'best.torchscript')
                shutil.copy(ts_path, ts_target_path)
                print(f"TorchScript模型已复制到: {ts_target_path}")
    except Exception as e:
        print(f"TorchScript导出失败: {e}")

    print("\n模型导出和保存操作完成！")
    
    # 清理内存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_yolo(use_augmentation=False, use_mixed_precision=False, config="default", resume=False):
    """
    YOLO训练配置，增加数据增强、混合精度训练和多种训练配置选项。
    支持CPU和GPU训练。
    Args:
        use_augmentation (bool): 是否启用数据增强，默认为False。
        use_mixed_precision (bool): 是否启用混合精度训练，默认为False。
        config (str): 训练配置模式，默认是'default'。可选项包括:
            - 'default': 默认配置
            - 'large_dataset': 数据集较大时的优化配置
            - 'small_dataset': 数据集较小时的优化配置
            - 'focus_accuracy': 注重检测精度时的优化配置
            - 'focus_speed': 注重训练速度时的优化配置
            - 'severmode': 使用租用服务器时的优化配置（完全覆盖原有配置） 
    """
    model = YOLO(select_model)  # 加载预训练的YOLO模型权重
    num_workers = max(1, min(os.cpu_count() - 2, 8))
    device = "cpu"
    if torch.cuda.is_available():
        device = "0"
    if device == "cpu":
        batch_size = 4  # 降低batch size
        workers = max(1, min(os.cpu_count() - 1, 4))  # 减少worker数量
        use_mixed_precision = False  # CPU不支持混合精度训练
    else:
        batch_size = 10  # 原始batch size
        workers = num_workers
    # 基础训练参数

    train_args = {
        "data": "data.yaml",  # 数据集配置文件路径
        "epochs": 120,  # 训练的总轮数
        "imgsz": 640,  # 输入图像的尺寸
        "batch": batch_size,  # 根据设备调整的批次大小
        "workers": workers,  # 根据设备调整的工作线程数
        "device": device,  # 自动选择的训练设备
        "patience": 15,  # 提前停止的容忍轮数
        "save_period": 5,  # 每隔多少轮保存一次模型
        "exist_ok": True,  # 如果结果目录存在，是否覆盖
        "project": os.path.dirname(os.path.abspath(__file__)),  # 训练结果的项目目录
        "name": "runs/train",  # 训练运行的名称
        "optimizer": "AdamW",  # 优化器类型
        "lr0": 0.0005,  # 初始学习率
        "lrf": 0.01,  # 最终学习率与初始学习率的比例
        "momentum": 0.937,  # 优化器的动量参数
        "weight_decay": 0.0005,  # 权重衰减（正则化）系数
        "warmup_epochs": 10,  # 预热阶段的轮数
        "warmup_momentum": 0.5,  # 预热阶段的动量
        "warmup_bias_lr": 0.05,  # 预热阶段偏置的学习率
        "box": 4.0,  # 边界框回归损失权重
        "cls": 2.0,  # 分类损失权重
        "dfl": 1.5,  # 分布式焦点损失权重
        "close_mosaic": 0,  # 是否关闭马赛克数据增强
        "nbs": 64,  # 基础批次大小
        "overlap_mask": False,  # 是否使用重叠掩码
        "multi_scale": True,  # 是否启用多尺度训练
        "single_cls": False,  # 是否将所有类别视为单一类别
        "rect": True,
        "cache": True,
    }
    train_args.update({"resume": resume})
    # 根据配置模式更新训练参数

    if config == "large_dataset":
        train_args.update(
            {
                "batch": 32 if device == "0" else 4,  # GPU使用32，CPU使用4
                "lr0": 0.001,
                "epochs": 150,
                "patience": 30,
            }
        )
    elif config == "small_dataset":
        train_args.update(
            {
                "batch": 16 if device == "0" else 4,  # GPU使用16，CPU使用4
                "lr0": 0.0001,
                "weight_decay": 0.001,
                "warmup_epochs": 15,
            }
        )
    elif config == "focus_accuracy":
        train_args.update(
            {
                "imgsz": 640,
                "box": 7.5,
                "cls": 4.0,
                "dfl": 3.0,
                "patience": 20,
                "batch": 16 if device == "0" else 4,  # GPU使用16，CPU使用4
                "epochs": 300,
                "lr0": 0.001,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "dropout": 0.1,# 添加dropout,减少过拟合
            }
        )
    elif config == "focus_speed":
        train_args.update(
            {
                "imgsz": 512,
                "epochs": 150,
                "patience": 30,
                "batch": 48 if device == "0" else 4,  # GPU使用48，CPU使用4
            }
        )
    elif config == "severmode":
        server_worker = int(os.cpu_count() / 2)
        train_args.update({
            # 保留focus_accuracy的精度优化参数
            "box": 7.5,                # 提高边界框损失权重
            "cls": 4.0,                # 提高分类损失权重
            "dfl": 3.0,                # 提高DFL损失权重
            "patience": 15,            # 保持高容忍度确保最佳精度
            "epochs": 150,             # 保持较长训练周期
            "dropout": 0.1,            # 保留dropout防止过拟合
            # 服务器性能优化参数
            "imgsz": 640,             
            "batch": 32,               # 提高batch size
            "lr0": 0.001,              # focus_accuracy的学习率
            "lrf": 0.01,               # focus_accuracy的学习率衰减
            "weight_decay": 0.0005,    # 保留weight_decay参数
            "optimizer": "AdamW",      # 使用AdamW优化器
            "workers": server_worker ,  # 充分利用CPU核心
            "device": "0",             # 确保使用GPU
            "half": True,              # 强制启用半精度训练
            "cache": "ram",            # 使用RAM缓存加速
            "cos_lr": True,            # 使用余弦学习率调度
            "warmup_epochs": 10,       # 保持充分的预热
            #"close_mosaic": 50,        # 这里有bug,先关闭
            "overlap_mask": True,      # 启用重叠掩码
            "save_period": 5,         # 定期保存检查点
            "multi_scale": True,       # 多尺度训练增强泛化能力
        })
        
    elif config != "default":
        print(f"警告: 未识别的配置模式 '{config}'，将使用默认配置。")
    # 数据增强参数
    if use_augmentation:
        augmentation_args = {
            "augment": True,  # 启用数据增强
            "degrees": 10.0,  # 随机旋转的角度范围
            "scale": 0.2,  # 随机缩放的比例范围
            "fliplr": 0.2,  # 随机水平翻转的概率
            "flipud": 0.2,  # 随机垂直翻转的概率
            #"hsv_h": 0.03,  # 随机调整色调的范围
            #"hsv_s": 0.2,  # 随机调整饱和度的范围
            #"hsv_v": 0.1,  # 随机调整明度的范围
            #"mosaic": 0.1,  # 马赛克增强的比例
            #"mixup": 0.1,  # 混合增强的比例
            #"copy_paste": 0.05,  # 复制粘贴增强的比例
        }
        train_args.update(augmentation_args)
    else:
        # 强制关闭所有数据增强
        no_augment_args = {
            "augment": False,  # 关闭数据增强
            "degrees": 0.0,  # 禁用旋转
            "scale": 0.0,  # 禁用缩放
            "fliplr": 0.0,  # 禁用水平翻转
            "flipud": 0.0,  # 禁用垂直翻转
            "hsv_h": 0.0,  # 禁用色调调整
            "hsv_s": 0.0,  # 禁用饱和度调整
            "hsv_v": 0.0,  # 禁用明度调整
            "mosaic": 0,  # 禁用马赛克
            "mixup": 0,  # 禁用mixup
            "copy_paste": 0,  # 禁用复制粘贴
        }
        train_args.update(no_augment_args)
    # 启用混合精度训练（仅在GPU上）

    if use_mixed_precision and device == "0":
        train_args.update({"half": True})
    else:
        train_args.update({"half": False})
    try:
        print(f"\n使用设备: {'GPU' if device == '0' else 'CPU'}")
        print(f"Batch size: {train_args['batch']}")
        print(f"混合精度训练: {'启用' if train_args.get('half', False) else '禁用'}\n")
        results = model.train(**train_args)
        
        # Force saving a best.pt if training was resumed
        if resume:
            print("\n检测到resume=True，确保保存最终模型到best.pt...")
            run_dir = results.save_dir if hasattr(results, 'save_dir') else train_args.get('project', '') + '/' + train_args.get('name', 'runs/train')
            weights_dir = os.path.join(run_dir, 'weights')
            last_pt_path = os.path.join(weights_dir, 'last.pt')
            best_pt_path = os.path.join(weights_dir, 'best.pt')
            
            if os.path.exists(last_pt_path):
                # 复制 last.pt 到 best.pt，确保总有最新的模型可用于后处理
                if not os.path.exists(best_pt_path) or resume:
                    print(f"将 {last_pt_path} 复制到 {best_pt_path}...")
                    shutil.copy2(last_pt_path, best_pt_path)
                    print(f"成功保存最终模型到 {best_pt_path}")
        
        return results
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def main():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        data_dir = datapath
        
        # 1. 检查数据集
        print("Step 1: Checking dataset...")
        valid_pairs = check_and_clean_dataset(data_dir)
        if not valid_pairs:
             print("No valid data pairs found. Exiting.")
             return # Exit if no data
        gc.collect()

        # 2. 创建配置文件
        print("\nStep 2: Creating data.yaml...")
        create_data_yaml()
        # Define data_yaml_path here for later use
        project_path = os.path.dirname(os.path.abspath(__file__))
        data_yaml_path = os.path.join(project_path, "data.yaml")

        # 3. 准备数据集 - 使用软链接
        print("\nStep 3: Preparing dataset with symbolic links...")
        try:
             # 启用符号链接选项
             train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs, use_symlinks=True)
             gc.collect()
             if val_size < 5: # Check validation set size after preparation
                 print(f"Warning: Validation set size ({val_size}) is less than 5. INT8 calibration might be suboptimal.")
        except ValueError as ve:
             print(f"Error during dataset preparation: {ve}")
             return # Exit if dataset prep fails critically
        
        print("\nStep 4: Starting training...")
        # Define your desired training config and resume flag
        training_config = "severmode" # Example: Use 'severmode' config
        resume_training = False       # Example: Start a new training run
        results = train_yolo(
            use_augmentation=True,      # Example: Enable augmentation
            use_mixed_precision=True,   # Example: Enable mixed precision (often handled by config)
            config=training_config,
            resume=resume_training
        )
        
        if results: # Check if training returned results (didn't fail)
            print("\nStep 5: Saving different precision models based on best.pt...")
            # Determine the path to the weights directory from the results object if possible
            # Fallback to constructing the path if needed
            weights_dir = None
            if hasattr(results, 'save_dir'):
                 weights_dir = os.path.join(results.save_dir, 'weights')
                 print(f"Found weights directory from results: {weights_dir}")
            else:
                 # Fallback: Construct the expected path (less reliable if name/project changed)
                 run_name = train_args.get('name', 'runs/train') # Get name from train_args if possible
                 project = train_args.get('project', project_path)
                 # Find the latest run directory based on the expected project/name structure
                 run_dirs = sorted(Path(project).glob(f"{Path(run_name).name}*/"), key=os.path.getmtime, reverse=True)
                 if run_dirs:
                     latest_run_dir = run_dirs[0]
                     weights_dir = os.path.join(latest_run_dir, 'weights')
                     print(f"Constructed weights directory: {weights_dir}")
                 else:
                     print(f"Warning: Could not determine weights directory automatically.")

            if weights_dir and os.path.isdir(weights_dir):
                 save_quantized_models(weights_dir, data_yaml_path)
            else:
                 print(f"Warning: Weights directory '{weights_dir}' not found. Skipping post-training save.")
        else:
            print("\nTraining did not complete successfully or was interrupted. Skipping post-training model saving.")
    except Exception as e:
        print(f"\nAn error occurred in the main execution flow: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
    finally: # Ensure final cleanup happens
        print("\nScript finished. Cleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()

