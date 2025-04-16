import os
import torch
import torchvision
import gc
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# 类别名称 - 保持与训练脚本一致
CLASS_NAMES = ["厨余垃圾", "可回收垃圾", "有害垃圾", "其他垃圾"]

# 选择要使用的模型类型，确保与训练时一致
MODEL_TYPE = "resnet50_fpn"  # 可选: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3"


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
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
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


def convert_checkpoint_to_final_model(checkpoint_path, output_dir, model_type=MODEL_TYPE):
    """将指定的检查点文件转换为最终模型"""
    try:
        print(f"开始将检查点 {checkpoint_path} 转换为最终模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 加载检查点
        print(f"正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 创建新模型
        print(f"创建模型架构 ({model_type})...")
        model = get_faster_rcnn_model(len(CLASS_NAMES), model_type=model_type)
        
        # 从检查点加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("成功加载模型参数")
        
        # 切换到评估模式
        model.eval()
        
        # 保存优化后的模型
        print("\n正在保存和优化模型用于部署...")
        model_path = save_optimized_model(model, output_dir, device, model_type)
        
        print(f"\n转换完成！模型已保存在 {output_dir} 目录中")
        print(f"最终模型路径: {model_path}")
        print("可以在树莓派上使用保存的ONNX或TorchScript模型进行部署。")
        
        return model_path
    
    except Exception as e:
        print(f"\n转换过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("\n脚本执行完毕。正在清理...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将训练中的检查点模型转换为可部署的最终模型')
    parser.add_argument('--model_path', type=str, default='./output/model_epoch_35.pth',
                        help='检查点文件路径 (默认: ./output/model_epoch_35.pth)')
    parser.add_argument('--out_dir', type=str, default='./output/final_model',
                        help='输出目录 (默认: ./output/final_model)')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE,
                        choices=['resnet50_fpn', 'resnet18_fpn', 'mobilenet_v3'],
                        help=f'模型类型 (默认: {MODEL_TYPE})')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"使用参数：")
    print(f"  - 模型路径: {args.model_path}")
    print(f"  - 输出目录: {args.out_dir}")
    print(f"  - 模型类型: {args.model_type}")
    
    # 执行转换
    convert_checkpoint_to_final_model(args.model_path, args.out_dir, args.model_type)
