#!/usr/bin/env python3
"""
Standalone program to convert YOLO .pt model to ONNX format
Usage:
    python convert_to_onnx.py --model path/to/model.pt [options]

Example:
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
    print(f"Error: Missing required dependencies - {e}")
    print("Please install dependencies first: pip install ultralytics tqdm onnx packaging")
    sys.exit(1)

def check_dependencies():
    """Check if dependency versions are compatible"""
    try:
        # Check ultralytics version
        ul_version = pkg_resources.get_distribution("ultralytics").version
        min_version = "8.0.0"
        if version.parse(ul_version) < version.parse(min_version):
            print(f"Warning: Current ultralytics version ({ul_version}) is older, recommended to upgrade to {min_version} or higher")
            print("You can run: pip install -U ultralytics\n")
        else:
            print(f"✓ ultralytics version {ul_version} is compatible")
            
        # Check onnx version
        onnx_version = pkg_resources.get_distribution("onnx").version
        min_onnx_version = "1.12.0"
        if version.parse(onnx_version) < version.parse(min_onnx_version):
            print(f"Warning: Current onnx version ({onnx_version}) is older, recommended to upgrade to {min_onnx_version} or higher")
            print("You can run: pip install -U onnx\n")
        else:
            print(f"✓ onnx version {onnx_version} is compatible")
            
        # Return version info for potential compatibility handling
        return ul_version, onnx_version
    except Exception as e:
        print(f"Version check error: {e}")
        return None, None

def print_welcome():
    """Print welcome message and usage instructions"""
    welcome_msg = """
╔════════════════════════════════════════════════╗
║        YOLO PT to ONNX Model Converter         ║
╚════════════════════════════════════════════════╝

This tool helps you convert YOLO PT models to ONNX format.

Basic usage:
  python convert_to_onnx.py --model your_model_path.pt

Common examples:
1. Basic conversion (with default parameters):
   python convert_to_onnx.py --model ./best.pt

2. Specify image size and enable FP16:
   python convert_to_onnx.py --model best.pt --imgsz 832 --half

3. Specify device:
   python convert_to_onnx.py --model best.pt --device cpu     # Use CPU
   python convert_to_onnx.py --model best.pt --device 0       # Use first GPU

4. Full parameter example:
   python convert_to_onnx.py --model best.pt --imgsz 640 --half --batch-size 4 --opset 11 --device 0

Available parameters:
  --model      : [Required] Path to input PT model file

  --imgsz      : [Optional] Input image size, default 640
                Effects:
                - Larger sizes (e.g., 832, 1024) improve detection accuracy, especially for small objects
                - Smaller sizes (e.g., 416, 512) improve inference speed
                - Very large sizes significantly increase memory usage and inference time
                Recommended values: 416/512/640/832/1024, must be multiples of 32

  --half       : [Optional] Enable FP16 half-precision export, default off
                Effects:
                - Reduces model size by ~50%
                - Improves inference speed by 30-50%
                - Slightly reduces accuracy (usually acceptable)
                - Requires hardware FP16 support

  --batch-size : [Optional] Batch size, default 1
                Effects:
                - Larger batches improve GPU utilization
                - Increases memory usage
                - If using --dynamic, this is the maximum batch size
                Recommended values: 1-32, adjust based on GPU memory

  --opset      : [Optional] ONNX opset version, default 12
                Effects:
                - Higher versions support more features
                - Lower versions have better compatibility
                - Affects which inference engine versions can deploy the model
                Recommended values: 11-13

  --no-simplify: [Optional] Disable ONNX model simplification, simplified by default
                Effects:
                - Disabling may preserve some original structures
                - Generally recommended to keep simplification enabled

  --no-dynamic : [Optional] Disable dynamic batch size, dynamic enabled by default
                Effects:
                - Disabling fixes batch size to --batch-size value
                - Suitable for deployment environments with fixed batch sizes

  --device     : [Optional] Specify device for conversion, auto-select by default
                Examples: --device cpu    # Use CPU
                         --device 0      # Use first GPU
                         --device 0,1    # Use multiple GPUs

  --validate   : [Optional] Validate ONNX model format after conversion

Performance optimization suggestions:
1. High accuracy scenario:
   --imgsz 832 --batch-size 1 --no-half --no-dynamic

2. Fast inference scenario:
   --imgsz 416 --batch-size 4 --half --device 0

3. Balanced configuration:
   --imgsz 640 --batch-size 1 --half --dynamic
"""
    print(welcome_msg)

def print_settings(args):
    """Print current parameter settings"""
    settings_msg = """
Current conversion settings:
╔════════════════════════════════════════════════╗
║ Input model: {:<37} ║
║ Image size: {:<37} ║
║ Precision: {:<37} ║
║ Batch size: {:<37} ║
║ ONNX version: {:<37} ║
║ Simplify model: {:<37} ║
║ Dynamic batch: {:<37} ║
║ Device: {:<37} ║
║ Validate model: {:<37} ║
╚════════════════════════════════════════════════╝
""".format(
    os.path.basename(args.model),
    f"{args.imgsz}x{args.imgsz}",
    "FP16" if args.half else "FP32",
    args.batch_size,
    f"OPSET {args.opset}",
    "Yes" if args.simplify else "No",
    "Yes" if args.dynamic else "No",
    args.device if args.device else "Auto-select",
    "Yes" if args.validate else "No"
)
    print(settings_msg)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO .pt model to ONNX format')
    
    # Required parameters
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the .pt model file')
    
    # Optional parameters                  
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
    
    # Set simplify and dynamic values (invert no-* parameters)
    args.simplify = not args.no_simplify
    args.dynamic = not args.no_dynamic
    
    return args

def find_onnx_model(model_path):
    """
    Find potentially generated ONNX model file without raising errors
    
    Args:
        model_path: Original PT model path
        
    Returns:
        str: ONNX model path if found, otherwise None
    """
    # Common export locations
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Possible file paths
    possible_paths = [
        os.path.join(model_dir, f"{model_name}.onnx"),  # Same directory, same name
        os.path.join(model_dir, "yolov8_onnx", f"{model_name}.onnx"),  # ultralytics default location
        os.path.join(os.getcwd(), f"{model_name}.onnx"),  # Current working directory
    ]
    
    # Find file
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def validate_onnx_model(model_path):
    """
    Validate ONNX model format
    
    Args:
        model_path: ONNX model path
        
    Returns:
        bool: Whether validation passed
    """
    try:
        print(f"\nValidating ONNX model: {model_path}")
        # Load and check model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # Get input/output info
        inputs = [input.name for input in onnx_model.graph.input]
        outputs = [output.name for output in onnx_model.graph.output]
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"✓ Model validation passed!")
        print(f"- Model size: {model_size:.2f} MB")
        print(f"- Input nodes: {', '.join(inputs)}")
        print(f"- Output nodes: {', '.join(outputs)}")
        print(f"- Operation nodes: {len(onnx_model.graph.node)}")
        
        return True
    except Exception as e:
        print(f"\n✗ Model validation failed: {str(e)}")
        return False

def convert_to_onnx(args):
    """
    Convert YOLO .pt model to ONNX format
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Whether conversion succeeded
    """
    try:
        # Check if input model file exists
        if not os.path.exists(args.model):
            print(f"Error: Model file not found {args.model}")
            return False
        
        # Print current settings
        print_settings(args)
        
        # Load model
        print("\n[1/3] Loading model...")
        start_time = time.time()
        try:
            model = YOLO(args.model)
        except Exception as e:
            print(f"Failed to load model, please ensure correct model format: {e}")
            return False
        
        load_time = time.time() - start_time
        print(f"Model loaded, time taken: {load_time:.2f} seconds")
        
        # Show basic model info
        try:
            model_info = f"Model type: {model.type}"
            print(f"Model info: {model_info}")
        except:
            print("Unable to get detailed model info")
        
        # Export to ONNX
        print("[2/3] Converting to ONNX format...")
        export_start = time.time()
        
        try:
            # Use progress bar to show conversion progress
            with tqdm(total=100, desc="Conversion progress", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                # Initial progress
                pbar.update(10)  # Initial preparation
                
                # Build command-line compatible arguments
                export_args = {
                    "format": "onnx",
                    "imgsz": args.imgsz,
                    "half": args.half,
                    "batch": args.batch_size,
                    "opset": args.opset,
                    "simplify": args.simplify,
                    "dynamic": args.dynamic
                }
                
                # Only add device if specified
                if args.device:
                    export_args["device"] = args.device
                    
                # Directly call export method
                success = model.export(**export_args)
                
                # Update progress bar
                pbar.update(90)  # Conversion complete
        except Exception as e:
            print(f"Error during conversion: {e}")
            return False
        
        export_time = time.time() - export_start
        
        # Try to find generated ONNX file
        onnx_path = find_onnx_model(args.model)
        
        if success:
            print(f"[3/3] Conversion complete! Time taken: {export_time:.2f} seconds")
            
            if onnx_path:
                print(f"\n✓ ONNX model generated: {onnx_path}")
                
                # Validate model format
                if args.validate:
                    validate_onnx_model(onnx_path)
                
                # Print model info
                print(f"\nFinal model info:")
                print(f"├── Conversion time: {export_time:.2f} seconds")
                print(f"├── Model size: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
                print(f"├── Input size: {args.imgsz}x{args.imgsz}")
                print(f"├── Batch size: {'Dynamic' if args.dynamic else args.batch_size}")
                print(f"├── Precision: {'FP16' if args.half else 'FP32'}")
                print(f"├── ONNX opset version: {args.opset}")
                print(f"└── Model simplified: {'Yes' if args.simplify else 'No'}")
                
                if args.half:
                    print("\nNote: You exported with FP16, please ensure your hardware supports FP16 inference")
                    
                if args.dynamic:
                    print("Note: You enabled dynamic batch size, can adjust batch size during deployment as needed")
            else:
                print("\n✓ Conversion appears successful but couldn't determine ONNX file location")
                print("Please check for generated file at:")
                model_dir = os.path.dirname(os.path.abspath(args.model))
                model_name = os.path.splitext(os.path.basename(args.model))[0]
                print(f"1. {os.path.join(model_dir, f'{model_name}.onnx')}")
                print(f"2. {os.path.join(os.getcwd(), f'{model_name}.onnx')}")
            
            return True
        else:
            print("\n✗ Conversion failed")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        return False

def main():
    # Print welcome message
    print_welcome()
    
    # Check dependency versions
    ul_version, onnx_version = check_dependencies()
    
    # Parse command line arguments
    args = parse_args()
    
    # Perform conversion
    result = convert_to_onnx(args)
    
    # Show final status
    if result:
        print("\nConversion completed successfully!")
        sys.exit(0)
    else:
        print("\nConversion failed. Please check error messages and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
