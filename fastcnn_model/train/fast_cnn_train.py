import os
import shutil
import json
import gc
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# 数据路径，根据实际情况修改
datapath = "./label"

# 选择模型类型
MODEL_TYPE = "resnet50_fpn"  # 标准版: "resnet50_fpn", 轻量版: "resnet18_fpn", 超轻量版: "mobilenet_v3"

# 四分类垃圾数据集配置
CATEGORY_MAPPING = {
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

# 类别名称
CLASS_NAMES = ["厨余垃圾", "可回收垃圾", "有害垃圾", "其他垃圾"]


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


def prepare_dataset(data_dir, valid_pairs):
    """准备数据集 - 修改验证集划分比例"""
    # 确保验证集至少有10张图片
    if len(valid_pairs) < 15:
        raise ValueError(
            f"有效数据对数量不足 ({len(valid_pairs)})。至少需要15张图片。"
        )
    
    # 清理现有目录
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
        os.makedirs(split_path, exist_ok=True)

    # 数据集划分 (90% 训练, 5% 验证, 5% 测试)
    print("将数据集划分为训练集、验证集和测试集...")
    train_files, temp = train_test_split(valid_pairs, test_size=0.1, random_state=42)
    val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)
    
    print("\n数据集准备完成。")
    print(f"训练集: {len(train_files)} 图片, 验证集: {len(val_files)} 图片, 测试集: {len(test_files)} 图片")
    
    return train_files, val_files, test_files


# 自定义数据集类
class GarbageDataset(Dataset):
    def __init__(self, img_files, data_dir, transforms=None):
        self.img_files = img_files
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(self.data_dir, base_name + ".json")
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        
        # 读取标注
        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        
        # 准备目标框和标签
        boxes = []
        labels = []
        for label in label_data.get("labels", []):
            if "name" not in label or label["name"] not in CATEGORY_MAPPING:
                continue
            
            # 检查边界框坐标是否存在
            required_keys = ["x1", "y1", "x2", "y2"]
            if not all(key in label for key in required_keys):
                continue
            
            # 获取类别ID和边界框坐标
            category_id = CATEGORY_MAPPING[label["name"]]
            x1, y1, x2, y2 = label["x1"], label["y1"], label["x2"], label["y2"]
            
            # 确保坐标有效
            x1, x2 = max(0, min(x1, image.shape[1])), max(0, min(x2, image.shape[1]))
            y1, y2 = max(0, min(y1, image.shape[0])), max(0, min(y2, image.shape[0]))
            
            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 跳过无效的框
            if width <= 0 or height <= 0:
                continue
            
            boxes.append([x1, y1, x2, y2])
            # TorchVision的目标检测模型需要从1开始的类别ID（0是背景）
            labels.append(category_id + 1)
        
        # 确保至少有一个目标
        if not boxes:
            # 创建一个很小的虚拟框和标签
            boxes = torch.as_tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.as_tensor([1], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 构建目标字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # 应用转换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


# 图像转换和增强
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
            
            # 更新边界框坐标
            width = image.shape[2]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
            
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def get_faster_rcnn_model(num_classes, model_type="resnet50_fpn"):
    """获取不同类型的Faster R-CNN模型"""
    # num_classes需要+1，因为0是背景类
    num_classes_with_bg = num_classes + 1
    
    if model_type == "resnet50_fpn":
        # 标准版：ResNet50+FPN
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    
    elif model_type == "resnet18_fpn":
        # 轻量版：ResNet18+FPN
        backbone = resnet_fpn_backbone(
            'resnet18', 
            pretrained=True, 
            trainable_layers=3
        )
        
        # 设置RPN的锚点生成器
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # 设置RoI池化的大小
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 创建Faster R-CNN模型
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes_with_bg,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    elif model_type == "mobilenet_v3":
        # 超轻量版：MobileNetV3
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """训练一个周期"""
    model.train()
    metric_logger = {}
    metric_logger["loss"] = 0
    metric_logger["loss_classifier"] = 0
    metric_logger["loss_box_reg"] = 0
    metric_logger["loss_objectness"] = 0
    metric_logger["loss_rpn_box_reg"] = 0
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_box = 0.0
    running_loss_obj = 0.0
    running_loss_rpn = 0.0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        # 计算损失
        loss_value = losses.item()
        loss_classifier = loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
        loss_box_reg = loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
        loss_objectness = loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
        
        running_loss += loss_value
        running_loss_cls += loss_classifier
        running_loss_box += loss_box_reg
        running_loss_obj += loss_objectness
        running_loss_rpn += loss_rpn_box_reg
        
        if not torch.isfinite(losses):
            print(f"警告: Loss不是有限值: {loss_value}")
            print(f"训练将继续，但请密切关注结果")
            # 跳过这个批次
            optimizer.zero_grad()
            continue
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if i % print_freq == 0 and i > 0:
            avg_loss = running_loss / print_freq
            avg_loss_cls = running_loss_cls / print_freq
            avg_loss_box = running_loss_box / print_freq
            avg_loss_obj = running_loss_obj / print_freq
            avg_loss_rpn = running_loss_rpn / print_freq
            
            elapsed = time.time() - start_time
            print(f"Epoch: {epoch} [{i}/{len(data_loader)}]\t"
                  f"Loss: {avg_loss:.4f}\t"
                  f"Time: {elapsed:.2f}s")
            
            # 重置计数器
            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_box = 0.0
            running_loss_obj = 0.0
            running_loss_rpn = 0.0
            start_time = time.time()
    
    # 计算周期的平均损失
    epoch_loss = metric_logger["loss"] / len(data_loader)
    return metric_logger


def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 推理
            model(images)
    
    # 简化版的评估，仅通过前向传播检查模型是否工作
    print("评估完成")
    return {}


def save_optimized_model(model, output_dir, device, model_type):
    """保存优化后的模型，包括不同格式和精度"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存基本模型
    model_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    
    # 导出模型（TorchScript格式）
    try:
        # 切换到评估模式
        model.eval()
        
        # 使用示例输入创建脚本模型
        dummy_input = [torch.rand(3, 640, 640).to(device)]
        script_model = torch.jit.trace(model, dummy_input)
        script_model_path = os.path.join(output_dir, "model_scripted.pt")
        torch.jit.save(script_model, script_model_path)
        print(f"TorchScript模型已保存至: {script_model_path}")
    except Exception as e:
        print(f"TorchScript导出失败: {e}")
    
    # 如果有CUDA，保存半精度模型
    if device != "cpu" and torch.cuda.is_available():
        try:
            model_fp16 = model.half()
            fp16_path = os.path.join(output_dir, "model_fp16.pth")
            torch.save(model_fp16.state_dict(), fp16_path)
            print(f"FP16模型已保存至: {fp16_path}")
            
            # 恢复为FP32
            model = model.float()
        except Exception as e:
            print(f"FP16模型保存失败: {e}")
    
    # 导出ONNX模型（适用于树莓派部署）
    try:
        dummy_input = torch.rand(1, 3, 640, 640).to(device)
        input_names = ["input"]
        output_names = ["boxes", "labels", "scores"]
        
        # 临时创建一个支持ONNX导出的前向函数
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, x):
                predictions = self.model([x])
                return (
                    predictions[0]["boxes"], 
                    predictions[0]["labels"], 
                    predictions[0]["scores"]
                )
        
        wrapper = ModelWrapper(model)
        
        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(
            wrapper, 
            dummy_input, 
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11
        )
        print(f"ONNX模型已保存至: {onnx_path}")
    except Exception as e:
        print(f"ONNX导出失败: {e}")
    
    # 模型量化（INT8）- 更适合在树莓派上运行
    try:
        # 注意：实际量化通常需要校准数据集
        # 这里仅展示一个简化的量化过程
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        q_path = os.path.join(output_dir, "model_quantized.pth")
        torch.save(quantized_model.state_dict(), q_path)
        print(f"量化模型已保存至: {q_path}")
    except Exception as e:
        print(f"量化模型保存失败: {e}")
    
    print("\n模型导出完成！")
    return model_path


def main():
    """主函数"""
    try:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. 检查数据集
        print("Step 1: 检查数据集...")
        valid_pairs = check_and_clean_dataset(datapath)
        if not valid_pairs:
            print("没有找到有效的数据对。退出。")
            return
        gc.collect()
        
        # 2. 准备数据集
        print("\nStep 2: 准备数据集...")
        try:
            train_files, val_files, test_files = prepare_dataset(datapath, valid_pairs)
            gc.collect()
            if len(val_files) < 5:
                print(f"警告: 验证集大小 ({len(val_files)}) 小于5。校准可能不理想。")
        except ValueError as ve:
            print(f"数据集准备过程中出错: {ve}")
            return
        
        # 3. 创建数据集和数据加载器
        print("\nStep 3: 创建数据集和数据加载器...")
        train_dataset = GarbageDataset(train_files, datapath, transforms=get_transform(train=True))
        val_dataset = GarbageDataset(val_files, datapath, transforms=get_transform(train=False))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 根据设备类型调整批次大小和工作线程
        if device.type == "cuda":
            batch_size = 8
            num_workers = min(6, os.cpu_count() // 2)
        else:
            batch_size = 2
            num_workers = min(2, os.cpu_count() // 2)
            # 对于CPU，始终确保至少有一个工作线程
            num_workers = max(1, num_workers)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # 4. 创建模型
        print(f"\nStep 4: 创建{MODEL_TYPE}模型...")
        model = get_faster_rcnn_model(len(CLASS_NAMES), model_type=MODEL_TYPE)
        model.to(device)
        
        # 5. 设置优化器和训练参数
        print("\nStep 5: 设置优化器和训练参数...")
        # 参数分组以应用不同的学习率
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, 
            lr=0.005, 
            momentum=0.9, 
            weight_decay=0.0005
        )
        
        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        
        # 训练轮数
        num_epochs = min(max(10, len(train_files) // 10), 50)
        print(f"将训练{num_epochs}个周期")
        
        # 6. 开始训练
        print("\nStep 6: 开始训练Faster R-CNN模型...")
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练一个周期
            train_metrics = train_one_epoch(
                model, optimizer, train_loader, device, epoch
            )
            
            # 更新学习率
            lr_scheduler.step()
            
            # 定期评估和保存
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                # 评估
                evaluate(model, val_loader, device)
                
                # 保存检查点
                checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)
                print(f"检查点已保存至: {checkpoint_path}")
        
        # 计算总训练时间
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n训练完成！总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        
        # 7. 保存最终模型和优化的部署版本
        print("\nStep 7: 保存和优化模型用于部署...")
        model_path = save_optimized_model(model, output_dir, device, MODEL_TYPE)
        
        print(f"\n训练完成！模型已保存在 {output_dir} 目录中。")
        print(f"最终模型路径: {model_path}")
        print("可以在树莓派上使用保存的ONNX或TorchScript模型进行部署。")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n脚本执行完毕。正在清理...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
