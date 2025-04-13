import os
import json
import argparse
from pathlib import Path

def convert_json_labels(input_json_path, output_json_path):
    """转换JSON标签到四大类垃圾分类"""
    try:
        if not os.path.exists(input_json_path):
            print(f"警告: JSON文件不存在: {input_json_path}")
            return False
        
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 类别映射
        category_mapping = {
            # 厨余垃圾 (0)
            "Kitchen waste": "Kitchen waste",
            "potato": "Kitchen waste",
            "daikon": "Kitchen waste",
            "carrot": "Kitchen waste",
            # 可回收垃圾 (1)
            "Recyclable waste": "Recyclable waste",
            "bottle": "Recyclable waste",
            "can": "Recyclable waste",
            # 有害垃圾 (2)
            "Hazardous waste": "Hazardous waste",
            "battery": "Hazardous waste",
            "drug": "Hazardous waste",
            "inner_packing": "Hazardous waste",
            # 其他垃圾 (3)
            "Other waste": "Other waste",
            "tile": "Other waste",
            "stone": "Other waste",
            "brick": "Other waste",
        }

        # 更新标签
        if "labels" in data:
            for label in data["labels"]:
                if "name" in label:
                    original_class = label["name"]
                    if original_class in category_mapping:
                        label["name"] = category_mapping[original_class]
                    else:
                        print(f"警告: 未知类别 {original_class} 在 {input_json_path}")
        
        # 保存转换后的JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式无效 {input_json_path}: {e}")
        return False
    except Exception as e:
        print(f"处理 {input_json_path} 时出错: {e}")
        return False

def batch_convert(input_dir, output_dir):
    """批量转换输入目录中的所有JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    total_files = len(json_files)
    success_count = 0
    
    print(f"找到 {total_files} 个JSON文件待处理")
    
    for idx, json_file in enumerate(json_files, 1):
        input_json_path = os.path.join(input_dir, json_file)
        output_json_path = os.path.join(output_dir, json_file)
        
        print(f"处理 [{idx}/{total_files}]: {json_file}")
        
        if convert_json_labels(input_json_path, output_json_path):
            success_count += 1
    
    print(f"\n转换完成: {success_count}/{total_files} 个文件成功转换")

def main():
    parser = argparse.ArgumentParser(description="将JSON标签转换为四大类垃圾分类标签")
    parser.add_argument("--input", "-i", type=str, default="./label", help="包含JSON标签文件的目录")
    parser.add_argument("--output", "-o", type=str, default="./labels_converted", help="转换后标签的输出目录")
    args = parser.parse_args()
    
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    
    batch_convert(args.input, args.output)

if __name__ == "__main__":
    main()
