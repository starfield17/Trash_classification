import os
import torch
import cv2
import numpy as np

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"


def find_camera(width=1280, height=720):
    """查找可用的摄像头并设置分辨率"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Set the resolution explicitly
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify if the resolution was actually set
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"成功找到可用摄像头，索引为: {index}")
            print(f"请求分辨率: {width}x{height}, 实际分辨率: {actual_width}x{actual_height}")
            
            return cap
        cap.release()
    print("错误: 未找到任何可用的摄像头")
    return None

def crop_frame(frame, target_width=720, target_height=720, mode='center', points=None):
    """
    裁切视频帧到指定尺寸
    
    参数:
        frame: 输入的视频帧（numpy数组，OpenCV图像格式）
            例如：frame = cv2.imread('image.jpg') 或从摄像头获取的帧
            
        target_width: 目标宽度，默认为720
            推荐值：根据模型输入要求或显示需求设置，常用值有224, 256, 320, 480, 640, 720
            例如：target_width=480 表示裁切后图像宽度为480像素
            
        target_height: 目标高度，默认为720
            推荐值：根据模型输入要求或显示需求设置，常用值有224, 256, 320, 480, 640, 720
            例如：target_height=480 表示裁切后图像高度为480像素
            
        mode: 裁切模式，可选值为：
            - 'center': 从中心裁切（默认）
                适用场景：当目标物体位于图像中央时使用，如正面人像、居中摆放的物品等
                用法：保留图像中央区域，裁剪掉周围区域
                示例：crop_frame(frame, 480, 480, mode='center')
                结果：从图像中央提取480x480的区域
                
            - 'left': 从左侧裁切
                适用场景：当目标物体位于图像左侧时使用，如向左侧靠近的物体
                用法：保留图像左侧区域，裁剪掉右侧区域
                示例：crop_frame(frame, 300, 720, mode='left')
                结果：从图像左侧提取宽300高720的区域，保留了图像的左侧部分
                
            - 'right': 从右侧裁切
                适用场景：当目标物体位于图像右侧时使用，如向右侧靠近的物体
                用法：保留图像右侧区域，裁剪掉左侧区域
                示例：crop_frame(frame, 300, 720, mode='right')
                结果：从图像右侧提取宽300高720的区域，保留了图像的右侧部分
                
            - 'top': 从顶部裁切
                适用场景：当目标物体位于图像上方时使用，如上方摆放的物品、俯视视角
                用法：保留图像上部区域，裁剪掉下部区域
                示例：crop_frame(frame, 720, 300, mode='top')
                结果：从图像顶部提取宽720高300的区域，保留了图像的上部分
                
            - 'bottom': 从底部裁切
                适用场景：当目标物体位于图像下方时使用，如桌面上的物品、仰视视角
                用法：保留图像下部区域，裁剪掉上部区域
                示例：crop_frame(frame, 720, 300, mode='bottom')
                结果：从图像底部提取宽720高300的区域，保留了图像的下部分
                
            - 'points': 使用四个点坐标进行裁切
                适用场景：需要裁切任意四边形区域时使用
                用法：提供四个点的坐标，函数将对这四个点围成的区域进行透视变换裁切
                示例：crop_frame(frame, 480, 480, mode='points', points=[(100,100), (500,50), (600,400), (50,450)])
                结果：将四个点围成的四边形区域变换为480x480的矩形图像
                
            注意：如果原始图像尺寸小于目标尺寸，函数会自动调整目标尺寸为较小的值
    返回:
        裁切后的视频帧
    """
    # 获取原始帧的尺寸
    frame_height, frame_width = frame.shape[:2]
    if mode == 'points':
        if points is None or len(points) != 4:
            raise ValueError("使用'points'模式时必须提供四个点坐标")
        src_points = np.array(points, dtype=np.float32)
        
        # （按左上、右上、右下、左下顺序）
        dst_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        cropped_frame = cv2.warpPerspective(frame, perspective_matrix, (target_width, target_height))
        
        return cropped_frame
    
    # 如果原始尺寸小于目标尺寸，调整目标尺寸为更小的值
    if frame_width < target_width or frame_height < target_height:
        print(f"警告: 原始帧尺寸({frame_width}x{frame_height})小于目标尺寸({target_width}x{target_height})，将调整目标尺寸")
        # 调整目标尺寸为原始尺寸和目标尺寸中较小的值
        target_width = min(frame_width, target_width)
        target_height = min(frame_height, target_height)
    
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
