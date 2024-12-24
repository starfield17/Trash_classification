#!/usr/bin/env python3
"""
YOLO .pt 模型转换为 ONNX 格式的独立程序
使用方法:
    python convert_to_onnx.py --model path/to/model.pt [options]

示例:
    python convert_to_onnx.py --model runs/train/weights/best.pt --imgsz 640 --half
"""

import argparse
import os
from ultralytics import YOLO

def print_welcome():
    """打印欢迎信息和使用说明"""
    welcome_msg = """
╔════════════════════════════════════════════════╗
║        YOLO PT to ONNX 模型转换工具            ║
╚════════════════════════════════════════════════╝

这个工具可以帮助你将YOLO的PT模型转换为ONNX格式。

基本用法:
  python convert_to_onnx.py --model 你的模型路径.pt

常用示例:
1. 基本转换（使用默认参数）:
   # 使用相对路径
   python convert_to_onnx.py --model ./best.pt
   python convert_to_onnx.py --model ../models/best.pt
   
   # 使用绝对路径
   # Linux/Mac:
   python convert_to_onnx.py --model /home/user/models/best.pt
   # Windows:
   python convert_to_onnx.py --model C:\\Models\\best.pt

2. 指定图像尺寸和启用FP16:
   python convert_to_onnx.py --model best.pt --imgsz 832 --half

3. 完整参数示例:
   python convert_to_onnx.py --model best.pt --imgsz 640 --half --batch-size 4 --opset 11 --output converted_model.onnx

可用参数说明:
  --model      : [必需] 输入的PT模型文件路径
                例如: --model ./best.pt                (相对路径，当前目录下的best.pt)
                      --model ../models/best.pt        (相对路径，上级目录models文件夹下的best.pt)
                      --model /home/user/best.pt       (Linux绝对路径)
                      --model C:\\Models\\best.pt      (Windows绝对路径)

  --imgsz      : [可选] 图像输入尺寸，默认640
                影响：
                - 较大的尺寸（如832、1024）可以提高检测精度，特别是对小目标
                - 较小的尺寸（如416、512）可以提高推理速度
                - 过大尺寸会显著增加内存占用和推理时间
                推荐值：416/512/640/832/1024，必须是32的倍数
                例如：--imgsz 832

  --half       : [可选] 启用FP16半精度导出，默认关闭
                影响：
                - 启用后模型大小减少约50%
                - 推理速度提升30-50%
                - 精度略有下降（通常可以接受）
                - 需要硬件支持FP16
                使用：--half（不需要额外参数）

  --batch-size : [可选] 批处理大小，默认1
                影响：
                - 较大的批次可以提高GPU利用率
                - 会增加内存占用
                - 如果使用--dynamic，此值为最大批次大小
                推荐值：1-32之间，根据显存大小调整
                例如：--batch-size 4

  --opset      : [可选] ONNX操作集版本，默认12
                影响：
                - 较高版本支持更多新特性
                - 较低版本兼容性更好
                - 影响模型可部署的推理引擎版本
                推荐值：11-13
                例如：--opset 11

  --simplify   : [可选] 简化ONNX模型，默认启用
                影响：
                - 启用后可减少模型大小
                - 优化模型结构，可能提升推理速度
                - 极少情况可能影响精度
                使用：--simplify 启用（默认）
                      --no-simplify 禁用

  --dynamic    : [可选] 启用动态批处理大小，默认启用
                影响：
                - 启用后可以在推理时动态调整批次大小
                - 增加部署灵活性
                - 可能略微影响推理速度
                使用：--dynamic 启用（默认）
                      --no-dynamic 禁用

  --output     : [可选] 输出文件路径，默认与输入文件同目录
                例如：--output ./converted/model.onnx
                      --output ../models/converted.onnx
                      如果不指定，将在输入模型同目录下生成同名的.onnx文件

性能优化建议:
1. 高精度场景:
   --imgsz 832 --batch-size 1 --no-half --simplify

2. 快速推理场景:
   --imgsz 416 --batch-size 4 --half --dynamic --simplify

3. 平衡配置:
   --imgsz 640 --batch-size 1 --half --dynamic --simplify
"""
    print(welcome_msg)

def print_settings(args):
    """打印当前设置的参数信息"""
    settings_msg = """
当前转换设置:
╔════════════════════════════════════════════════╗
║ 输入模型: {:<37} ║
║ 输出路径: {:<37} ║
║ 图像尺寸: {:<37} ║
║ 精度模式: {:<37} ║
║ 批处理数: {:<37} ║
║ ONNX版本: {:<37} ║
║ 模型简化: {:<37} ║
║ 动态批次: {:<37} ║
╚════════════════════════════════════════════════╝
""".format(
    os.path.basename(args.model),
    os.path.basename(args.output) if args.output else "自动生成",
    f"{args.imgsz}x{args.imgsz}",
    "FP16" if args.half else "FP32",
    args.batch_size,
    f"OPSET {args.opset}",
    "是" if args.simplify else "否",
    "是" if args.dynamic else "否"
)
    print(settings_msg)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO .pt model to ONNX format')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the .pt model file')
    
    # 可选参数                  
    parser.add_argument('--imgsz', type=int, default=640,
                      help='Image size (default: 640)')
    parser.add_argument('--half', action='store_true',
                      help='Enable FP16 half-precision export')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size (default: 1)')
    parser.add_argument('--opset', type=int, default=12,
                      help='ONNX opset version (default: 12)')
    parser.add_argument('--simplify', action='store_true', default=True,
                      help='Enable ONNX model simplification')
    parser.add_argument('--dynamic', action='store_true', default=True,
                      help='Enable dynamic batch size')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path (default: same directory as input with .onnx extension)')
    
    return parser.parse_args()

def convert_to_onnx(args):
    """
    将YOLO的.pt模型转换为ONNX格式
    
    Args:
        args: 命令行参数
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 检查输入模型文件是否存在
        if not os.path.exists(args.model):
            print(f"错误: 未找到模型文件 {args.model}")
            return False
            
        # 如果未指定输出路径，使用默认路径
        if args.output is None:
            args.output = os.path.splitext(args.model)[0] + '.onnx'
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # 打印当前设置
        print_settings(args)
        
        # 加载模型
        print("\n[1/3] 正在加载模型...")
        model = YOLO(args.model)
        
        # 导出为ONNX
        print("[2/3] 正在转换为ONNX格式...")
        success = model.export(
            format="onnx",
            imgsz=args.imgsz,
            half=args.half,
            batch=args.batch_size,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic,
            save=True
        )
        
        if success:
            print("[3/3] 转换完成！")
            print(f"\n✓ ONNX模型已保存至: {args.output}")
            # 打印模型信息
            print("\n最终模型信息:")
            print(f"├── 输入尺寸: {args.imgsz}x{args.imgsz}")
            print(f"├── 批处理大小: {'dynamic' if args.dynamic else args.batch_size}")
            print(f"├── 精度: {'FP16' if args.half else 'FP32'}")
            print(f"├── ONNX操作集版本: {args.opset}")
            print(f"└── 模型简化: {'是' if args.simplify else '否'}")
            return True
        else:
            print("\n✗ 转换失败")
            return False
            
    except Exception as e:
        print(f"\n✗ 转换过程中出错: {str(e)}")
        return False

def main():
    # 打印欢迎信息
    print_welcome()
    
    # 解析命令行参数
    args = parse_args()
    
    # 执行转换
    convert_to_onnx(args)

if __name__ == "__main__":
    main()
