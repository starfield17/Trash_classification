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
import sys
import time
from pathlib import Path
import pkg_resources
from packaging import version

try:
    from ultralytics import YOLO
    from tqdm import tqdm
    import onnx
except ImportError as e:
    print(f"错误: 缺少必要的依赖 - {e}")
    print("请先安装依赖: pip install ultralytics tqdm onnx packaging")
    sys.exit(1)

def check_dependencies():
    """检查依赖版本是否兼容"""
    try:
        # 检查ultralytics版本
        ul_version = pkg_resources.get_distribution("ultralytics").version
        min_version = "8.0.0"
        if version.parse(ul_version) < version.parse(min_version):
            print(f"警告: 当前 ultralytics 版本 ({ul_version}) 较旧，建议升级到 {min_version} 或更高")
            print("您可以运行: pip install -U ultralytics\n")
        else:
            print(f"✓ ultralytics 版本 {ul_version} 兼容")
            
        # 检查 onnx 版本
        onnx_version = pkg_resources.get_distribution("onnx").version
        min_onnx_version = "1.12.0"
        if version.parse(onnx_version) < version.parse(min_onnx_version):
            print(f"警告: 当前 onnx 版本 ({onnx_version}) 较旧，建议升级到 {min_onnx_version} 或更高")
            print("您可以运行: pip install -U onnx\n")
        else:
            print(f"✓ onnx 版本 {onnx_version} 兼容")
            
        # 返回版本信息用于后续可能的兼容性处理
        return ul_version, onnx_version
    except Exception as e:
        print(f"版本检查出错: {e}")
        return None, None

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
   python convert_to_onnx.py --model ./best.pt

2. 指定图像尺寸和启用FP16:
   python convert_to_onnx.py --model best.pt --imgsz 832 --half

3. 指定设备:
   python convert_to_onnx.py --model best.pt --device cpu     # 使用CPU
   python convert_to_onnx.py --model best.pt --device 0       # 使用第一张GPU

4. 完整参数示例:
   python convert_to_onnx.py --model best.pt --imgsz 640 --half --batch-size 4 --opset 11 --device 0

可用参数说明:
  --model      : [必需] 输入的PT模型文件路径

  --imgsz      : [可选] 图像输入尺寸，默认640
                影响：
                - 较大的尺寸（如832、1024）可以提高检测精度，特别是对小目标
                - 较小的尺寸（如416、512）可以提高推理速度
                - 过大尺寸会显著增加内存占用和推理时间
                推荐值：416/512/640/832/1024，必须是32的倍数

  --half       : [可选] 启用FP16半精度导出，默认关闭
                影响：
                - 启用后模型大小减少约50%
                - 推理速度提升30-50%
                - 精度略有下降（通常可以接受）
                - 需要硬件支持FP16

  --batch-size : [可选] 批处理大小，默认1
                影响：
                - 较大的批次可以提高GPU利用率
                - 会增加内存占用
                - 如果使用--dynamic，此值为最大批次大小
                推荐值：1-32之间，根据显存大小调整

  --opset      : [可选] ONNX操作集版本，默认12
                影响：
                - 较高版本支持更多新特性
                - 较低版本兼容性更好
                - 影响模型可部署的推理引擎版本
                推荐值：11-13

  --no-simplify: [可选] 禁用ONNX模型简化，默认启用简化
                影响：
                - 禁用后可能保留一些原始结构
                - 通常情况下建议保持简化

  --no-dynamic : [可选] 禁用动态批处理大小，默认启用动态
                影响：
                - 禁用后批处理大小将固定为--batch-size值
                - 适用于批处理大小固定的部署环境

  --device     : [可选] 指定转换使用的设备，默认自动选择
                例如：--device cpu    # 使用CPU
                      --device 0      # 使用第一张GPU
                      --device 0,1    # 使用多张GPU

  --validate   : [可选] 转换完成后验证ONNX模型格式

性能优化建议:
1. 高精度场景:
   --imgsz 832 --batch-size 1 --no-half --no-dynamic

2. 快速推理场景:
   --imgsz 416 --batch-size 4 --half --device 0

3. 平衡配置:
   --imgsz 640 --batch-size 1 --half --dynamic
"""
    print(welcome_msg)

def print_settings(args):
    """打印当前设置的参数信息"""
    settings_msg = """
当前转换设置:
╔════════════════════════════════════════════════╗
║ 输入模型: {:<37} ║
║ 图像尺寸: {:<37} ║
║ 精度模式: {:<37} ║
║ 批处理数: {:<37} ║
║ ONNX版本: {:<37} ║
║ 模型简化: {:<37} ║
║ 动态批次: {:<37} ║
║ 使用设备: {:<37} ║
║ 验证模型: {:<37} ║
╚════════════════════════════════════════════════╝
""".format(
    os.path.basename(args.model),
    f"{args.imgsz}x{args.imgsz}",
    "FP16" if args.half else "FP32",
    args.batch_size,
    f"OPSET {args.opset}",
    "是" if args.simplify else "否",
    "是" if args.dynamic else "否",
    args.device if args.device else "自动选择",
    "是" if args.validate else "否"
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
    parser.add_argument('--no-simplify', action='store_true',
                      help='Disable ONNX model simplification')
    parser.add_argument('--no-dynamic', action='store_true',
                      help='Disable dynamic batch size')
    parser.add_argument('--device', type=str, default='',
                      help='Device to use (e.g., cpu, 0, 0,1)')
    parser.add_argument('--validate', action='store_true',
                      help='Validate ONNX model after conversion')
    
    args = parser.parse_args()
    
    # 设置 simplify 和 dynamic 的值（反转 no-* 参数）
    args.simplify = not args.no_simplify
    args.dynamic = not args.no_dynamic
    
    return args

def find_onnx_model(model_path):
    """
    查找可能生成的ONNX模型文件，但不抛出错误
    
    Args:
        model_path: 原始PT模型路径
        
    Returns:
        str: ONNX模型路径，如果找不到则返回None
    """
    # 常见的导出位置
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # 可能的文件路径
    possible_paths = [
        os.path.join(model_dir, f"{model_name}.onnx"),  # 同目录同名
        os.path.join(model_dir, "yolov8_onnx", f"{model_name}.onnx"),  # ultralytics默认位置
        os.path.join(os.getcwd(), f"{model_name}.onnx"),  # 当前工作目录
    ]
    
    # 查找文件
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def validate_onnx_model(model_path):
    """
    验证ONNX模型格式是否正确
    
    Args:
        model_path: ONNX模型路径
        
    Returns:
        bool: 验证是否通过
    """
    try:
        print(f"\n正在验证ONNX模型: {model_path}")
        # 加载并检查模型
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # 获取输入输出信息
        inputs = [input.name for input in onnx_model.graph.input]
        outputs = [output.name for output in onnx_model.graph.output]
        
        # 获取模型大小
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"✓ 模型验证通过！")
        print(f"- 模型大小: {model_size:.2f} MB")
        print(f"- 输入节点: {', '.join(inputs)}")
        print(f"- 输出节点: {', '.join(outputs)}")
        print(f"- 操作节点数: {len(onnx_model.graph.node)}")
        
        return True
    except Exception as e:
        print(f"\n✗ 模型验证失败: {str(e)}")
        return False

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
        
        # 打印当前设置
        print_settings(args)
        
        # 加载模型
        print("\n[1/3] 正在加载模型...")
        start_time = time.time()
        try:
            model = YOLO(args.model)
        except Exception as e:
            print(f"加载模型失败，请确保模型格式正确: {e}")
            return False
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        
        # 显示模型基本信息
        try:
            model_info = f"模型类型: {model.type}"
            print(f"模型信息: {model_info}")
        except:
            print("无法获取模型详细信息")
        
        # 导出为ONNX
        print("[2/3] 正在转换为ONNX格式...")
        export_start = time.time()
        
        try:
            # 使用进度条提示转换进度
            with tqdm(total=100, desc="转换进度", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                # 初始进度
                pbar.update(10)  # 初始准备阶段
                
                # 构建为命令行兼容格式的参数
                export_args = {
                    "format": "onnx",
                    "imgsz": args.imgsz,
                    "half": args.half,
                    "batch": args.batch_size,
                    "opset": args.opset,
                    "simplify": args.simplify,
                    "dynamic": args.dynamic
                }
                
                # 只有当设备参数不为空时才添加
                if args.device:
                    export_args["device"] = args.device
                    
                # 直接调用export方法
                success = model.export(**export_args)
                
                # 更新进度条
                pbar.update(90)  # 转换完成
        except Exception as e:
            print(f"转换过程中出错: {e}")
            return False
        
        export_time = time.time() - export_start
        
        # 尝试找到生成的ONNX文件
        onnx_path = find_onnx_model(args.model)
        
        if success:
            print(f"[3/3] 转换完成！耗时: {export_time:.2f} 秒")
            
            if onnx_path:
                print(f"\n✓ ONNX模型已生成: {onnx_path}")
                
                # 验证模型格式
                if args.validate:
                    validate_onnx_model(onnx_path)
                
                # 打印模型信息
                print(f"\n最终模型信息:")
                print(f"├── 转换耗时: {export_time:.2f} 秒")
                print(f"├── 模型大小: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
                print(f"├── 输入尺寸: {args.imgsz}x{args.imgsz}")
                print(f"├── 批处理大小: {'动态' if args.dynamic else args.batch_size}")
                print(f"├── 精度: {'FP16' if args.half else 'FP32'}")
                print(f"├── ONNX操作集版本: {args.opset}")
                print(f"└── 模型简化: {'是' if args.simplify else '否'}")
                
                if args.half:
                    print("\n提示: 您已使用FP16导出模型，请确保您的硬件支持FP16推理")
                    
                if args.dynamic:
                    print("提示: 您已使用动态批处理大小，部署时可根据需要调整批处理大小")
            else:
                print("\n✓ 转换过程似乎成功，但无法确定ONNX文件的位置")
                print("请在以下位置查找生成的文件:")
                model_dir = os.path.dirname(os.path.abspath(args.model))
                model_name = os.path.splitext(os.path.basename(args.model))[0]
                print(f"1. {os.path.join(model_dir, f'{model_name}.onnx')}")
                print(f"2. {os.path.join(os.getcwd(), f'{model_name}.onnx')}")
            
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
    
    # 检查依赖版本
    ul_version, onnx_version = check_dependencies()
    
    # 解析命令行参数
    args = parse_args()
    
    # 执行转换
    result = convert_to_onnx(args)
    
    # 显示最终状态
    if result:
        print("\n转换过程成功完成!")
        sys.exit(0)
    else:
        print("\n转换过程失败。请检查错误信息并重试。")
        sys.exit(1)

if __name__ == "__main__":
    main()
