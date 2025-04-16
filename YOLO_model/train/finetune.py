import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained YOLOv model')
    parser.add_argument('--pretrained', type=str, required=True, 
                        help='Path to the pre-trained best.pt model')
    parser.add_argument('--data-path', type=str, default='./finetune_data',
                        help='Path to the new data for fine-tuning')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for fine-tuning (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (default: 0.0001)')
    return parser.parse_args()

def validate_json_file(json_path):
    """验证 JSON 文件是否有效"""
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

def _check_single_file(img_file, data_dir):
    """检查单个图片文件及其对应的JSON标签文件"""
    img_path = os.path.join(data_dir, img_file)
    base_name = os.path.splitext(img_file)[0]
    json_file = os.path.join(data_dir, base_name + ".json")

    # 检查图片完整性
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 损坏或无效的图片文件: {img_file}")
            return None
        # 检查图片尺寸是否合理
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            print(f"警告: 图片尺寸过小: {img_file}")
            return None
    except Exception as e:
        print(f"警告: 读取图片 {img_file} 时出错: {e}")
        return None

    # 检查标签文件是否存在且有效
    if os.path.exists(json_file):
        # 确保标签文件确实是一个 JSON 文件
        if not json_file.lower().endswith(".json"):
            print(f"警告: 标签文件扩展名不正确 (应为 .json): {json_file}")
            return None

        # 验证 JSON 文件内容
        if not validate_json_file(json_file):
            return None

        # 验证 JSON 文件结构
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            if "labels" not in label_data:
                print(f"警告: 标签文件结构无效 (缺少 'labels' 键): {json_file}")
                return None
            # 如果所有检查都通过，返回图像文件名
            return img_file
        except Exception as e:
            print(f"警告: 处理标签文件 {json_file} (结构检查) 时发生错误: {e}")
            return None
    else:
        print(f"警告: 找不到对应的标签文件: {json_file}")
        return None

def check_and_clean_dataset(data_dir):
    """检查数据集完整性并清理无效数据"""
    print("正在检查数据集完整性...")
    image_extensions = (".jpg", ".jpeg", ".png")

    try:
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
    # 确定最大工作线程数
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"使用最多 {max_workers} 个 worker 线程进行检查...")

    # 使用ThreadPoolExecutor进行并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个图像文件提交检查任务
        for img_file in image_files:
            futures.append(executor.submit(_check_single_file, img_file, data_dir))

        # 收集结果
        for future in futures:
            try:
                result = future.result()
                if result:
                    valid_pairs.append(result)
            except Exception as exc:
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
        "nc": 4,  # 4个大类
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    
    return os.path.join(current_dir, "data.yaml")

def try_create_symlink(src, dst):
    """尝试创建符号链接，成功返回True，失败返回False"""
    try:
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception as e:
        print(f"警告: 无法创建从 {src} 到 {dst} 的符号链接: {e}")
        return False

def process_split(split_name, files, data_dir, use_symlinks=True):
    """处理数据集划分的图像复制/链接和标签转换"""
    print(f"\n处理 {split_name} 划分...")
    split_img_dir = os.path.join(split_name, "images")
    split_lbl_dir = os.path.join(split_name, "labels")

    # 确保目标目录存在
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)
    
    # 测试是否可以使用符号链接
    symlinks_working = False
    if use_symlinks and files:
        test_img = os.path.join(data_dir, files[0])
        test_dst = os.path.join(split_img_dir, "test_symlink_" + files[0])
        symlinks_working = try_create_symlink(test_img, test_dst)
        
        # 清理测试链接
        if os.path.exists(test_dst):
            os.remove(test_dst)
    
    print(f"{split_name}: 使用{'符号链接' if symlinks_working else '文件复制'}进行数据集准备")

    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(data_dir, img_file)
        src_json = os.path.join(data_dir, base_name + ".json")
        dst_img = os.path.join(split_img_dir, img_file)
        dst_txt = os.path.join(split_lbl_dir, base_name + ".txt")

        # 处理图像和标签文件
        if os.path.exists(src_img) and os.path.exists(src_json):
            try:
                # 处理图像文件 - 如果启用尝试先使用符号链接
                if symlinks_working:
                    if not try_create_symlink(src_img, dst_img):
                        # 如果创建符号链接失败，回退到复制
                        shutil.copy2(src_img, dst_img)
                else:
                    # 如果符号链接不可用，直接复制
                    shutil.copy2(src_img, dst_img)
                
                # 转换标签
                convert_labels(src_json, dst_txt)
            except Exception as e:
                print(f"处理文件对({img_file}, {base_name}.json)时出错: {e}")
        else:
            if not os.path.exists(src_img):
                print(f"警告: 在分割处理期间未找到源图像: {src_img}")
            if not os.path.exists(src_json):
                print(f"警告: 在分割处理期间未找到源JSON: {src_json}")

    print(f"{split_name}: 处理了 {len(files)} 个潜在图像")

def prepare_dataset(data_dir, valid_pairs, use_symlinks=True):
    """准备数据集 - 划分为训练/验证/测试集"""
    # 确保验证集至少有5张图片
    if len(valid_pairs) < 10:
        raise ValueError(
            f"有效数据对数量不足 ({len(valid_pairs)})。至少需要10张图片以便微调。"
        )
    
    # 清理现有目录
    print("\n正在清理现有的train/val/test目录...")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), split)
        if os.path.exists(split_path):
            print(f"正在删除 {split_path} 目录...")
            try:
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
                        except Exception as e:
                            print(f"警告: 无法删除文件 {file_path}: {e}")
                    
                    # 再删除空目录
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            print(f"警告: 无法删除目录 {dir_path}: {e}")
                
                # 最后尝试删除主目录
                try:
                    os.rmdir(split_path)
                except Exception as e:
                    print(f"警告: 无法删除主目录 {split_path}: {e}")
        
        # 重新创建目录结构
        print(f"创建 {split} 目录结构...")
        for subdir in ["images", "labels"]:
            subdir_path = os.path.join(split_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)

    # 对于微调，我们使用较多的验证数据以确保模型的泛化能力
    # 数据集划分 (70% 训练, 20% 验证, 10% 测试)
    print("将数据集划分为训练集、验证集和测试集...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.33, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # 使用ThreadPoolExecutor并行处理每个数据集分割
    print("开始并行处理数据集划分...")
    max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_split, split_name, files, data_dir, use_symlinks): split_name
            for split_name, files in splits.items()
        }

        # 等待所有任务完成并检查异常
        for future in futures:
            split_name = futures[future]
            try:
                future.result()
                print(f"完成处理 {split_name} 划分。")
            except Exception as exc:
                print(f'{split_name} 划分生成了一个异常: {exc}')

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
            print(f"警告: 未找到JSON文件: {json_file}")
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
            print(f"警告: 未找到对应的图像文件: {json_file}")
            return False
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
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
                print(f"警告: JSON文件 {json_file} 中没有'labels'键")
                return False
                
            for label in data["labels"]:
                try:
                    if "name" not in label:
                        print(f"警告: JSON文件 {json_file} 中的标签数据没有'name'字段")
                        continue
                        
                    class_name = label["name"]
                    if class_name not in category_mapping:
                        print(f"警告: 未知类别 {class_name} 在 {json_file} 中")
                        continue
                        
                    # 直接使用大分类ID
                    category_id = category_mapping[class_name]

                    required_keys = ["x1", "y1", "x2", "y2"]
                    if not all(key in label for key in required_keys):
                        print(f"警告: 在 {json_file} 中缺少边界框坐标")
                        continue
                        
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        label, img_width, img_height
                    )

                    if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                        f.write(
                            f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )
                    else:
                        print(f"警告: 在 {json_file} 中的边界框值无效")
                        continue
                except KeyError as e:
                    print(f"警告: 在 {json_file} 中的标签数据缺少键: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 处理 {json_file} 中的标签时出错: {e}")
                    continue
        return True
    except json.JSONDecodeError as e:
        print(f"错误: {json_file} 中的JSON格式无效: {e}")
        return False
    except Exception as e:
        print(f"处理 {json_file} 时出错: {e}")
        return False

def finetune_yolo(pretrained_model, data_yaml_path, args):
    """
    对预训练的YOLOv模型进行微调
    
    Args:
        pretrained_model: 预训练模型路径
        data_yaml_path: 数据配置文件路径
        args: 命令行参数
    """
    print(f"\n正在加载预训练模型: {pretrained_model}")
    try:
        model = YOLO(pretrained_model)
    except Exception as e:
        print(f"加载预训练模型时出错: {e}")
        return None
        
    # 确定设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "0"
        
    # 调整微调的批次大小
    if device == "cpu":
        batch_size = min(4, args.batch_size)  # CPU上限制批次大小
        workers = max(1, min(os.cpu_count() - 1, 4))
        use_half = False  # CPU不支持混合精度训练
    else:
        batch_size = args.batch_size
        workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        use_half = True  # GPU上使用混合精度加速训练
        
    # 配置微调参数
    finetune_args = {
        "data": data_yaml_path,
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": batch_size,
        "workers": workers,
        "device": device,
        "patience": args.patience,
        "save_period": 5,
        "exist_ok": True,
        "project": os.path.dirname(os.path.abspath(__file__)),
        "name": "runs/finetune",
        "optimizer": "AdamW",
        "lr0": args.lr,  # 微调时使用较小的学习率
        "lrf": 0.01,  # 学习率衰减因子
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,  # 微调时减少预热轮数
        "warmup_momentum": 0.5,
        "warmup_bias_lr": 0.05,
        "box": 7.5,  # 增加边界框权重以提高定位准确度
        "cls": 4.0,  # 增加分类权重
        "dfl": 3.0,
        "nbs": 64,
        "val": True,  # 确保在每个epoch后进行验证
        "rect": True,  # 使用矩形训练提高效率
        "cache": True,
        "single_cls": False,
        "half": use_half,  # 根据设备设置是否使用半精度
        # 数据增强设置
        "augment": True,
        "degrees": 5.0,  # 微调时减少旋转角度
        "scale": 0.1,  # 微调时减少缩放范围
        "fliplr": 0.5,  # 增加水平翻转概率
        "hsv_h": 0.015,  # 减少色调变化
        "hsv_s": 0.1,  # 减少饱和度变化
        "hsv_v": 0.05,  # 减少亮度变化
        "mosaic": 0.5,  # 使用马赛克增强
        "mixup": 0.1,  # 保留mixup但降低强度
    }
    
    # 开始微调
    try:
        print(f"\n使用设备: {'GPU' if device == '0' else 'CPU'}")
        print(f"批次大小: {finetune_args['batch']}")
        print(f"学习率: {finetune_args['lr0']}")
        print(f"训练轮数: {finetune_args['epochs']}")
        print(f"混合精度训练: {'启用' if finetune_args['half'] else '禁用'}\n")
        
        results = model.train(**finetune_args)
        
        # 返回微调结果
        print("\n微调完成！")
        return results
    except Exception as e:
        print(f"微调时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_finetuned_model(model_dir):
    """
    保存微调后的模型，并创建一个模型信息文件
    
    Args:
        model_dir: 微调模型的保存目录
    """
    best_pt_path = os.path.join(model_dir, 'weights', 'best.pt')
    
    if not os.path.exists(best_pt_path):
        print(f"错误: 微调后的模型 {best_pt_path} 未找到")
        return
        
    # 创建一个信息文件，记录微调信息
    info_path = os.path.join(model_dir, 'finetuned_model_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("微调模型信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型路径: {best_pt_path}\n")
        f.write(f"微调时间: {Path(best_pt_path).stat().st_mtime}\n")
        f.write(f"模型大小: {Path(best_pt_path).stat().st_size / (1024*1024):.2f} MB\n")
        f.write("微调模型配置:\n")
        f.write("- 使用了预训练权重进行微调\n")
        f.write("- 适用于垃圾分类四类检测 (厨余垃圾、可回收垃圾、有害垃圾、其他垃圾)\n")
        
    print(f"\n微调模型已保存至: {best_pt_path}")
    print(f"模型信息已保存至: {info_path}")
    
    # 复制一个带版本号的备份
    version = 1
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuned_models')
    os.makedirs(backup_dir, exist_ok=True)
    
    # 查找当前最高版本
    existing_models = list(Path(backup_dir).glob('finetuned_v*.pt'))
    if existing_models:
        versions = [int(m.stem.split('_v')[1]) for m in existing_models]
        version = max(versions) + 1
        
    backup_path = os.path.join(backup_dir, f'finetuned_v{version}.pt')
    shutil.copy2(best_pt_path, backup_path)
    print(f"备份模型已保存至: {backup_path}")

def main():
    """主函数 - 微调预训练的YOLOv模型"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 检查预训练模型是否存在
        if not os.path.exists(args.pretrained):
            print(f"错误: 预训练模型 {args.pretrained} 不存在")
            return
            
        data_dir = args.data_path
        
        # 1. 检查数据集
        print("步骤 1: 检查数据集...")
        valid_pairs = check_and_clean_dataset(data_dir)
        if not valid_pairs:
            print("未找到有效的数据对。退出。")
            return
        gc.collect()
        
        # 2. 创建配置文件
        print("\n步骤 2: 创建data.yaml...")
        data_yaml_path = create_data_yaml()
        
        # 3. 准备数据集
        print("\n步骤 3: 准备数据集...")
        try:
            train_size, val_size, test_size = prepare_dataset(data_dir, valid_pairs, use_symlinks=True)
            gc.collect()
            if val_size < 3:
                print(f"警告: 验证集大小 ({val_size}) 小于3。微调可能不会产生良好结果。")
        except ValueError as ve:
            print(f"数据集准备过程中出错: {ve}")
            return
            
        # 4. 微调模型
        print("\n步骤 4: 开始微调...")
        results = finetune_yolo(args.pretrained, data_yaml_path, args)
        
        # 5. 保存微调后的模型
        if results:
            print("\n步骤 5: 保存微调后的模型...")
            # 如果results有save_dir属性，使用它作为模型目录
            if hasattr(results, 'save_dir'):
                model_dir = results.save_dir
            else:
                # 否则构造期望的路径
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/finetune')
                
            save_finetuned_model(model_dir)
        else:
            print("\n微调未成功完成或被中断。跳过模型保存。")
            
    except Exception as e:
        print(f"\n主执行流程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n脚本结束。正在清理...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
