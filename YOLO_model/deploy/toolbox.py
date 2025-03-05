import os
import torch
import cv2


def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"


def find_camera():
    """查找可用的摄像头"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功找到可用摄像头，索引为: {index}")
            return cap
        cap.release()
    print("错误: 未找到任何可用的摄像头")
    return None

def crop_frame(frame, target_width=720, target_height=720, mode='center'):
    """
    裁切视频帧到指定尺寸
    
    参数:
        frame: 输入的视频帧（numpy数组，OpenCV图像格式）
        target_width: 目标宽度，默认为720
        target_height: 目标高度，默认为720
        mode: 裁切模式，可选值为：
            - 'center': 从中心裁切（默认）
            - 'left': 从左侧裁切
            - 'right': 从右侧裁切
            - 'top': 从顶部裁切
            - 'bottom': 从底部裁切
    
    返回:
        裁切后的视频帧
    """
    import cv2
    import numpy as np
    
    # 获取原始帧的尺寸
    frame_height, frame_width = frame.shape[:2]
    
    # 检查原始尺寸是否小于目标尺寸
    if frame_width < target_width or frame_height < target_height:
        print(f"警告: 原始帧尺寸({frame_width}x{frame_height})小于目标尺寸({target_width}x{target_height})，将进行缩放")
        # 计算缩放比例
        scale = max(target_width / frame_width, target_height / frame_height)
        # 缩放图像
        frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
        # 更新尺寸
        frame_height, frame_width = frame.shape[:2]
    
    # 计算裁切区域
    if mode == 'center':
        # 从中心裁切
        start_x = (frame_width - target_width) // 2
        start_y = (frame_height - target_height) // 2
    elif mode == 'left':
        # 从左侧裁切
        start_x = 0
        start_y = (frame_height - target_height) // 2
    elif mode == 'right':
        # 从右侧裁切
        start_x = frame_width - target_width
        start_y = (frame_height - target_height) // 2
    elif mode == 'top':
        # 从顶部裁切
        start_x = (frame_width - target_width) // 2
        start_y = 0
    elif mode == 'bottom':
        # 从底部裁切
        start_x = (frame_width - target_width) // 2
        start_y = frame_height - target_height
    else:
        raise ValueError(f"不支持的裁切模式: {mode}")
    
    # 确保起始坐标不为负
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    
    # 确保不超出图像边界
    if start_x + target_width > frame_width:
        start_x = frame_width - target_width
    if start_y + target_height > frame_height:
        start_y = frame_height - target_height
    
    # 裁切图像
    cropped_frame = frame[start_y:start_y+target_height, start_x:start_x+target_width]
    
    return cropped_frame

def get_script_directory():
    script_path = os.path.abspath(__file__)
    directory = os.path.dirname(script_path)
    print(f"脚本目录: {directory}")
    return directory


class WasteClassifier:
    def __init__(self):
        # 分类名称

        self.class_names = {
            0: "厨余垃圾",
            1: "可回收垃圾",
            2: "有害垃圾",
            3: "其他垃圾",
        }

        # 细分类到大分类的映射 - 移除,因为我们直接使用四大类

        self.category_mapping = None

        # 分类名称对应的描述(可选)

        self.category_descriptions = {
            0: "厨余垃圾",
            1: "可回收利用垃圾",
            2: "有害垃圾",
            3: "其他垃圾",
        }

    def get_category_info(self, class_id):
        """
        获取给定类别ID的分类信息
        返回: (分类名称, 分类描述)
        """
        category_name = self.class_names.get(class_id, "未知分类")
        description = self.category_descriptions.get(class_id, "未知描述")

        return category_name, description

    def print_classification(self, class_id):
        """打印分类信息"""
        category_name, description = self.get_category_info(class_id)
        print(f"\n垃圾分类信息:")
        print(f"分类类别: {category_name}")
        print(f"分类说明: {description}")
        print("-" * 30)

        return f"{category_name}"
