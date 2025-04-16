# fastcnn_autolabel.py
import os
import cv2
import json
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

# 导入PyTorch和Faster R-CNN所需的库
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision import transforms as T
from torchvision.transforms import functional as F

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fastcnn_autolabeling.log')
    ]
)
logger = logging.getLogger(__name__)

class FastCNNAutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.5, model_type="resnet50_fpn", iou_threshold=0.5):
        """初始化基于Faster R-CNN的自动标注器

        Args:
            model_path: Faster R-CNN模型文件路径(.pth)
            confidence_threshold: 保留检测结果的最小置信度
            model_type: 使用的骨干网络类型("resnet50_fpn", "resnet18_fpn", 或 "mobilenet_v3")
            iou_threshold: 用于NMS的IoU阈值，超过该值的框被认为重叠
        """
        # 保存IoU阈值
        self.iou_threshold = iou_threshold
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.confidence_threshold = confidence_threshold
            self.model_type = model_type
            
            # 定义类别名称(从fastcnn_deploy.py中获取)
            self.model_class_names = ["厨余垃圾", "可回收垃圾", "有害垃圾", "其他垃圾"]
            
            # 加载模型
            self._load_model(model_path)
            
            logger.info(f"已加载Faster R-CNN模型: {model_path}")
            logger.info(f"模型类型: {model_type}")
            logger.info(f"模型类别名称: {self.model_class_names}")
            logger.info(f"使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"无法加载Faster R-CNN模型 {model_path}: {e}")
            raise  # 重新抛出异常以在模型加载失败时停止执行
        
        # 定义输出类别名称和从模型类ID的映射
        self.category_names = {
            0: "Kitchen_waste",     # 厨余垃圾
            1: "Recyclable_waste",  # 可回收垃圾
            2: "Hazardous_waste",   # 有害垃圾
            3: "Other_waste",       # 其他垃圾
        }
        
        # 定义预处理转换
        self.transforms = T.Compose([
            T.ToTensor(),
        ])
        
        # 创建反向映射和颜色
        self.category_mapping = {v: k for k, v in self.category_names.items()}
        self.category_colors = {
            0: (86, 180, 233),    # 厨余垃圾 - 蓝色
            1: (230, 159, 0),     # 可回收垃圾 - 橙色
            2: (240, 39, 32),     # 有害垃圾 - 红色
            3: (0, 158, 115),     # 其他垃圾 - 绿色
        }
        
        logger.info(f"FastCNNAutoLabeler初始化完成.")
        logger.info(f"置信度阈值: {self.confidence_threshold}")
        logger.info(f"输出类别映射: {self.category_names}")
    
    def _get_faster_rcnn_model(self, num_classes, model_type):
        """获取不同类型的Faster R-CNN模型
        
        Args:
            num_classes: 模型中的类别数
            model_type: 使用的骨干网络类型("resnet50_fpn", "resnet18_fpn", 或 "mobilenet_v3")
            
        Returns:
            FasterRCNN: Faster R-CNN模型
        """
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
            
            # 设置RPN的锚点生成器 - 修改这里以匹配多个特征图层
            anchor_generator = AnchorGenerator(
                # 为每个特征图层指定单独的锚点尺寸
                sizes=((32,), (64,), (128,), (256,), (512,)),
                # 为每个特征图层重复相同的比例配置
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )
            
            # 设置RoI池化的大小
            roi_pooler = torch.ops.torchvision.MultiScaleRoIAlign(
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
    
    def _load_model(self, model_path):
        """加载Faster R-CNN模型
        
        Args:
            model_path: 模型权重文件路径
        """
        try:
            # 获取类别数量
            num_classes = len(self.model_class_names)
            
            # 基于模型类型构建网络
            self.model = self._get_faster_rcnn_model(num_classes, self.model_type)
            
            # 加载预训练权重
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理可能的键名差异
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"尝试直接加载模型失败，尝试检查点格式: {e}")
                # 尝试加载检查点格式 (如果是带有 model_state_dict 的检查点)
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    logger.info("成功从检查点加载权重")
                else:
                    raise RuntimeError("无法加载模型权重，请检查模型文件格式")
            
            # 将模型移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_image(self, img_path, output_dir, viz_dir=None):
        """使用Faster R-CNN处理单张图像并创建JSON标签文件

        Args:
            img_path: 输入图像路径
            output_dir: 保存JSON标签文件的目录
            viz_dir: 保存可视化结果的目录(如果为None，则不保存可视化)

        Returns:
            tuple: (结果消息, 状态, 检测数量)
        """
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                return f"无法读取图像: {img_path}", "failed", 0

            # 获取图像尺寸
            img_height, img_width = img.shape[:2]

            # 创建JSON文件的输出路径
            output_path = output_dir / f"{img_path.stem}.json"

            # 检查JSON文件是否已存在
            if output_path.exists():
                return f"跳过(已存在): {output_path}", "skipped", 0

            # 调用Faster R-CNN模型获取检测结果
            detections = self.detect_objects(img)

            # 保存JSON文件
            if not detections:
                self.save_empty_json(output_path)
                result_msg = f"无有效检测: {img_path.name}"
                detection_count = 0
            else:
                self.save_json(output_path, detections, img_width, img_height)
                result_msg = f"已创建 {output_path.name} 包含 {len(detections)} 个检测结果"
                detection_count = len(detections)

            # 如果需要，生成可视化
            if viz_dir is not None:
                viz_path = viz_dir / f"{img_path.stem}_labeled.jpg"
                self.visualize_detections(img, detections, viz_path)

            return result_msg, "success", detection_count

        except Exception as e:
            logger.exception(f"处理 {img_path.name} 时出错")
            return f"处理 {img_path.name} 时出错: {str(e)}", "failed", 0
    
    def _calculate_iou(self, boxA, boxB):
        """计算两个边界框的IoU (Intersection over Union)

        Args:
            boxA: 第一个边界框 [x1, y1, x2, y2]
            boxB: 第二个边界框 [x1, y1, x2, y2]

        Returns:
            float: IoU值，范围[0, 1]
        """
        # 确定交集矩形
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # 计算交集面积
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # 计算两个边界框的面积
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # 计算IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _apply_nms(self, detections, iou_threshold=0.5):
        """应用非极大值抑制(NMS)，保留重叠框中置信度最高的

        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值，超过该值认为两个框重叠

        Returns:
            list: 应用NMS后的检测结果列表
        """
        if len(detections) == 0:
            return []

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        selected_dets = []

        while sorted_dets:
            # 选择置信度最高的检测
            best_det = sorted_dets.pop(0)
            selected_dets.append(best_det)

            # 过滤掉与当前最佳检测重叠的框
            remaining_dets = []
            for det in sorted_dets:
                box1 = [best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"]]
                box2 = [det["x1"], det["y1"], det["x2"], det["y2"]]
                iou = self._calculate_iou(box1, box2)

                # 如果IoU小于阈值，则保留该检测
                if iou < iou_threshold:
                    remaining_dets.append(det)

            sorted_dets = remaining_dets

        return selected_dets

    def detect_objects(self, img):
        """使用加载的Faster R-CNN模型检测对象

        Args:
            img: 图像(NumPy数组)

        Returns:
            list: 格式化保存/可视化的有效检测结果列表
        """
        valid_detections = []
        try:
            # 转换BGR为RGB(Faster R-CNN模型需要RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            image_tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # 从预测中提取检测结果
            if len(predictions) > 0:
                prediction = predictions[0]
                
                # 获取预测的边界框、分数和标签
                boxes = prediction['boxes'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                
                # 处理检测结果
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    # 检查置信度阈值
                    if score < self.confidence_threshold:
                        continue
                    
                    # 记住label从1开始(0是背景)，所以实际类别是label-1
                    class_id = int(label) - 1
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 将检测到的类ID映射到所需的类别名称
                    category_name = self.category_names.get(class_id)
                    
                    if category_name is not None:
                        # 存储检测结果
                        valid_detections.append({
                            "name": category_name,  # 使用映射的名称
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": float(score)
                        })
                    else:
                        logger.debug(f"跳过未知类ID检测: {class_id} (置信度: {score:.2f})")
                
                # 应用NMS，只保留重叠框中置信度最高的
                valid_detections = self._apply_nms(valid_detections, iou_threshold=self.iou_threshold)
                logger.debug(f"应用NMS后的检测数量: {len(valid_detections)}")
            
        except Exception as e:
            logger.error(f"Faster R-CNN检测过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        return valid_detections
    
    def save_empty_json(self, output_path):
        """保存没有检测结果的空JSON文件"""
        data = {"labels": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_json(self, output_path, detections, img_width, img_height):
        """将检测结果以所需格式保存到JSON文件中

        Args:
            output_path: 保存JSON文件的路径
            detections: 从detect_objects获取的检测对象列表
            img_width: 图像宽度(像素)
            img_height: 图像高度(像素)
        """
        labels = []
        for det in detections:
            # 基本验证和边界检查
            try:
                x1 = max(0, int(det.get('x1', 0)))
                y1 = max(0, int(det.get('y1', 0)))
                x2 = min(img_width, int(det.get('x2', img_width)))
                y2 = min(img_height, int(det.get('y2', img_height)))

                # 跳过无效的边界框
                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"跳过无效边界框: ({x1}, {y1}, {x2}, {y2})")
                    continue

                label = {
                    # 按原样保存名称(应为映射的类别名称)
                    # 若下游任务需要，则转换为小写
                    "name": det.get('name', 'unknown').lower(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(det.get('confidence', 0.0))
                }
                labels.append(label)
            except (ValueError, TypeError) as e:
                logger.warning(f"处理检测结果为JSON时出错: {e}")
                continue

        # 创建最终JSON结构
        data = {"labels": labels}

        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def visualize_detections(self, img, detections, output_path):
        """生成检测结果的可视化

        Args:
            img: 输入图像(NumPy数组)
            detections: 从detect_objects获取的检测对象列表
            output_path: 保存可视化的路径
        """
        vis_img = img.copy()

        for det in detections:
            name = det.get('name', 'unknown')  # 这是类别名称, 例如, "Kitchen_waste"
            x1 = max(0, int(det.get('x1', 0)))
            y1 = max(0, int(det.get('y1', 0)))
            x2 = min(vis_img.shape[1], int(det.get('x2', vis_img.shape[1])))
            y2 = min(vis_img.shape[0], int(det.get('y2', vis_img.shape[0])))

            if x2 <= x1 or y2 <= y1:
                continue

            # 使用类别名称获取类别ID和颜色
            category_id = self.category_mapping.get(name)  # 从名称查找ID
            if category_id is None:
                logger.warning(f"可视化过程中发现无效类别名称 '{name}'")
                color = (255, 255, 255)  # 默认白色
            else:
                color = self.category_colors.get(category_id, (255, 255, 255))

            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            confidence = det.get('confidence', 0.0)
            label_text = f"{name} {confidence:.2f}"

            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y_start = max(y1 - label_h - 10, 0)
            label_y_end = max(y1, label_h + 10)
            text_y = max(y1 - 5, label_h + 5)

            cv2.rectangle(vis_img, (x1, label_y_start), (x1 + label_w + 10, label_y_end), color, -1)

            # 绘制标签文本
            cv2.putText(vis_img, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 保存可视化图像
        if not cv2.imwrite(str(output_path), vis_img):
            logger.error(f"保存可视化图像失败: {output_path}")
    
    def verify_json_files(self, directory):
        """验证目录中所有JSON文件的格式是否正确

        Args:
            directory: 包含JSON文件的目录

        Returns:
            dict: 验证结果
        """
        json_files = list(Path(directory).glob('*.json'))
        results = {'valid': 0, 'invalid': 0, 'errors': []}
        # 获取有效名称(小写)用于检查
        valid_names_lower = {name.lower() for name in self.category_mapping.keys()}

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'labels' not in data or not isinstance(data['labels'], list):
                    results['invalid'] += 1
                    results['errors'].append((str(json_file), "缺少'labels'数组"))
                    continue

                invalid_labels = []
                for i, label in enumerate(data['labels']):
                    # 检查必需的键(name, x1, y1, x2, y2, confidence)
                    required_keys = ['name', 'x1', 'y1', 'x2', 'y2', 'confidence']
                    if not all(key in label for key in required_keys):
                        missing = [k for k in required_keys if k not in label]
                        invalid_labels.append((i, f"缺少必需字段: {missing}"))
                        continue

                    # 检查名称(小写)是否为有效的类别名称
                    if label.get('name') not in valid_names_lower:
                        invalid_labels.append((i, f"无效类别名称: {label.get('name')}"))
                        continue

                    # 可选: 添加坐标验证
                    try:
                        x1, y1, x2, y2 = int(label['x1']), int(label['y1']), int(label['x2']), int(label['y2'])
                        # 基本检查，假设不知道图像尺寸，只检查坐标关系
                        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                            invalid_labels.append((i, f"无效坐标: ({x1},{y1},{x2},{y2})"))
                    except (ValueError, TypeError):
                        invalid_labels.append((i, f"非整数坐标"))
                    # 可选: 检查置信度值
                    try:
                        conf = float(label['confidence'])
                        if not (0.0 <= conf <= 1.0):
                            invalid_labels.append((i, f"无效置信度值: {conf}"))
                    except (ValueError, TypeError):
                        invalid_labels.append((i, f"非浮点置信度值"))

                if invalid_labels:
                    results['invalid'] += 1
                    errors_str = ", ".join([f"标签 {idx}: {err}" for idx, err in invalid_labels])
                    results['errors'].append((str(json_file), f"{len(invalid_labels)} 个无效标签 ({errors_str})"))
                else:
                    results['valid'] += 1

            except json.JSONDecodeError:
                results['invalid'] += 1
                results['errors'].append((str(json_file), "无效JSON格式"))
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append((str(json_file), f"读取/解析错误: {str(e)}"))

        return results


def main():
    """主函数，处理命令行参数并运行标注器"""
    parser = argparse.ArgumentParser(description='使用本地Faster R-CNN模型自动标注图像')
    parser.add_argument('--input_dir', required=True, help='包含未标注图像的目录')
    parser.add_argument('--model_path', required=True, help='训练好的Faster R-CNN模型路径(.pth文件)')
    parser.add_argument('--model_type', default='resnet50_fpn', choices=['resnet50_fpn', 'resnet18_fpn', 'mobilenet_v3'], 
                        help='使用的骨干网络类型')
    parser.add_argument('--batch_size', type=int, default=4, help='并行处理的图像数量')
    parser.add_argument('--confidence', type=float, default=0.5, help='检测的最小置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='NMS的IoU阈值，用于过滤重叠框')
    parser.add_argument('--extensions', default='.jpg,.jpeg,.png,.bmp,.webp', help='图像扩展名的逗号分隔列表')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 验证模型路径
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error(f"未找到模型文件: {args.model_path}")
        return
    if model_path.suffix != ".pth":
        logger.warning(f"模型文件 {args.model_path} 没有.pth扩展名。请确保它是有效的Faster R-CNN模型。")

    # 创建标注器实例
    try:
        labeler = FastCNNAutoLabeler(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            model_type=args.model_type,
            iou_threshold=args.iou_threshold
        )
    except Exception as e:
        logger.error(f"初始化FastCNNAutoLabeler失败: {e}")
        return  # 如果初始化失败则退出

    # 处理输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"未找到输入目录: {args.input_dir}")
        return

    # --- 硬编码输出和可视化目录 ---
    base_dir = Path('.')  # 当前工作目录
    output_dir = base_dir / "output"  # 固定输出目录名
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- 可视化始终启用 ---
    viz_dir = base_dir / "viz"  # 固定可视化目录名
    viz_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"可视化结果将保存到 {viz_dir}")
    # --- 目录和可视化更改结束 ---

    # 获取图像文件列表
    image_extensions = args.extensions.lower().split(',')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))  # 同时检查大写扩展名

    # 以防万一，删除重复项
    image_files = list(set(image_files))

    if not image_files:
        logger.warning(f"在 {input_dir} 中未找到扩展名为 {args.extensions} 的图像文件")
        return

    logger.info(f"找到 {len(image_files)} 张图像待处理")

    # 批量处理图像
    successful = 0
    failed = 0
    skipped = 0
    empty = 0

    # 为所有图像创建future
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = []
        for img_path in image_files:
            futures.append(executor.submit(
                labeler.process_image,
                img_path,
                output_dir,
                viz_dir  # 直接传递viz_dir(现在始终启用)
            ))

        # 使用进度条处理
        with tqdm(total=len(image_files), desc="标注图像") as pbar:
            for future in futures:
                try:
                    result, status, detection_count = future.result()

                    if status == "success":
                        successful += 1
                        if detection_count == 0:
                            empty += 1
                        pbar.set_postfix({
                            "成功": successful, "空": empty, "跳过": skipped, "失败": failed
                        })
                    elif status == "skipped":
                        skipped += 1
                        pbar.set_postfix({
                            "成功": successful, "空": empty, "跳过": skipped, "失败": failed
                        })
                    else:  # status == "failed"
                        failed += 1
                        pbar.set_postfix({
                            "成功": successful, "空": empty, "跳过": skipped, "失败": failed
                        })
                        logger.error(f"失败任务: {result}")  # 记录失败任务的错误消息

                    if status != "failed":
                        logger.info(result)  # 记录成功/跳过消息
                    pbar.update(1)

                except Exception as e:
                    # 捕获future.result()本身的异常
                    logger.error(f"从线程检索结果时出错: {e}")
                    failed += 1
                    pbar.set_postfix({
                        "成功": successful, "空": empty, "跳过": skipped, "失败": failed
                    })
                    pbar.update(1)

    # --- 验证始终启用 ---
    logger.info("验证生成的JSON文件...")
    verification_results = labeler.verify_json_files(output_dir)
    logger.info(f"验证结果: {verification_results['valid']} 个有效, {verification_results['invalid']} 个无效")
    if verification_results['errors']:
        logger.warning(f"发现 {verification_results['invalid']} 个无效文件。")
        # 记录前几个错误以供检查
        errors_to_show = min(5, len(verification_results['errors']))
        logger.info(f"前 {errors_to_show} 个验证错误:")
        for i, (file, error) in enumerate(verification_results['errors'][:errors_to_show]):
            logger.info(f"  文件: {file} | 错误: {error}")
    else:
        logger.info("所有生成的JSON文件通过验证。")
    # --- 验证更改结束 ---

    logger.info(f"处理完成:")
    logger.info(f"  - 总图像数: {len(image_files)}")
    logger.info(f"  - 成功: {successful} ({empty} 个无检测结果)")
    logger.info(f"  - 跳过(已标注): {skipped}")
    logger.info(f"  - 失败: {failed}")


if __name__ == "__main__":
    main()
