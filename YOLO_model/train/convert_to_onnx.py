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
import shutil
from pathlib import Path
import onnx
import pkg_resources
from packaging import version

try:
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_version
    from tqdm import tqdm
except ImportError as e:
    print(f"错误: 缺少必要的依赖 - {e}")
    print("请先安装依赖: pip install ultralytics tqdm packaging")
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

3. 指定设备:
   python convert_to_onnx.py --model best.pt --device cpu     # 使用CPU
   python convert_to_onnx.py --model best.pt --device 0       # 使用第一张GPU
   python convert_to_onnx.py --model best.pt --device 0,1     # 使用多张GPU

4. 完整参数示例:
   python convert_to_onnx.py --model best.pt --imgsz 640 --half --batch-size 4 --opset 11 --output converted_model.onnx --device 0

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

  --no-simplify: [可选] 禁用ONNX模型简化，默认启用简化
                影响：
                - 禁用后可能保留一些原始结构
                - 通常情况下建议保持简化
                使用：--no-simplify（不需要额外参数）

  --no-dynamic : [可选] 禁用动态批处理大小，默认启用动态
                影响：
                - 禁用后批处理大小将固定为--batch-size值
                - 适用于批处理大小固定的部署环境
                使用：--no-dynamic（不需要额外参数）

  --output     : [可选] 输出文件路径，默认与输入文件同目录
                例如：--output ./converted/model.onnx
                      --output ../models/converted.onnx
                      如果不指定，将在输入模型同目录下生成同名的.onnx文件

  --device     : [可选] 指定转换使用的设备，默认自动选择
                例如：--device cpu    # 使用CPU
                      --device 0      # 使用第一张GPU
                      --device 0,1    # 使用前两张GPU

  --validate   : [可选] 转换完成后验证ONNX模型格式
                使用：--validate（不需要额外参数）

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
║ 输出路径: {:<37} ║
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
    os.path.basename(args.output) if args.output else "自动生成",
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
    parser.add_argument('--output', type=str, default=None,
                      help='Output path (default: same directory as input with .onnx extension)')
    parser.add_argument('--device', type=str, default='',
                      help='Device to use (e.g., cpu, 0, 0,1)')
    parser.add_argument('--validate', action='store_true',
                      help='Validate ONNX model after conversion')
    
    args = parser.parse_args()
    
    # 设置 simplify 和 dynamic 的值（反转 no-* 参数）
    args.simplify = not args.no_simplify
    args.dynamic = not args.no_dynamic
    
    return args

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
            
        # 如果未指定输出路径，使用默认路径
        if args.output is None:
            args.output = os.path.splitext(args.model)[0] + '.onnx'
            
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:  # 如果目录不为空
            os.makedirs(output_dir, exist_ok=True)
        
        # 打印当前设置
        print_settings(args)
        
        # 加载模型
        print("\n[1/4] 正在加载模型...")
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
        print("[2/4] 正在转换为ONNX格式...")
        export_start = time.time()
        
        try:
            # 使用进度条提示转换进度
            with tqdm(total=100, desc="转换进度", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                # 模拟进度更新
                def update_progress(progress):
                    pbar.update(progress - pbar.n)
                
                # 进度更新函数
                def progress_callback(current, total):
                    progress = int((current / total) * 100)
                    update_progress(progress)
                
                # 预设几个进度点
                update_progress(10)  # 初始准备阶段
                
                success = model.export(
                    format="onnx",
                    imgsz=args.imgsz,
                    half=args.half,
                    batch=args.batch_size,
                    opset=args.opset,
                    simplify=args.simplify,
                    dynamic=args.dynamic,
                    device=args.device,
                    file=args.output  # 明确指定输出文件
                )
                
                update_progress(90)  # 转换基本完成
                time.sleep(0.5)  # 给一点时间显示
                update_progress(100)  # 完成
        except Exception as e:
            print(f"转换过程中出错: {e}")
            return False
            
        export_time = time.time() - export_start
        
        # 验证模型格式
        if args.validate and success:
            print("[3/4] 正在验证ONNX模型...")
            validation_success = validate_onnx_model(args.output)
            if not validation_success:
                print("警告: 模型验证失败，但文件已生成")
        else:
            print("[3/4] 跳过模型验证...")
        
        if success:
            print("[4/4] 转换完成！")
            print(f"\n✓ ONNX模型已保存至: {args.output}")
            # 打印模型信息
            print(f"\n最终模型信息:")
            print(f"├── 转换耗时: {export_time:.2f} 秒")
            print(f"├── 模型大小: {os.path.getsize(args.output) / (1024 * 1024):.2f} MB")
            print(f"├── 输入尺寸: {args.imgsz}x{args.imgsz}")
            print(f"├── 批处理大小: {'动态' if args.dynamic else args.batch_size}")
            print(f"├── 精度: {'FP16' if args.half else 'FP32'}")
            print(f"├── ONNX操作集版本: {args.opset}")
            print(f"└── 模型简化: {'是' if args.simplify else '否'}")
            
            if args.half:
                print("\n提示: 您已使用FP16导出模型，请确保您的硬件支持FP16推理")
                
            if args.dynamic:
                print("提示: 您已使用动态批处理大小，部署时可根据需要调整批处理大小")
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
