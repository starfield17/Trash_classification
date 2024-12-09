import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 全局控制变量
DEBUG_WINDOW = True
CONF_THRESHOLD = 0.5  # 置信度阈值

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 类别名称和颜色映射
        self.class_names = {
            0: 'battery',
            1: 'expired_drug',
            2: 'inner_packing',
            3: 'can',
            4: 'bottle',
            5: 'potato',
            6: 'daikon',
            7: 'carrot',
            8: 'tile',
            9: 'pebble',
            10: 'brick'
        }
        
        # 为每个类别生成随机颜色
        self.colors = {}
        np.random.seed(42)  # 固定随机种子以保持颜色一致
        for class_id in self.class_names:
            self.colors[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))

    def detect(self, frame):
        """对单帧图像进行检测"""
        # 使用YOLOv8进行检测
        results = self.model(frame, conf=CONF_THRESHOLD)
        
        # 处理检测结果
        if len(results) > 0:
            result = results[0]  # 获取第一个图像的结果
            
            # 获取所有检测框
            boxes = result.boxes
            
            # 在图像上绘制检测结果
            for box in boxes:
                # 获取坐标和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                # 获取类别名称和颜色
                class_name = self.class_names.get(class_id, "unknown")
                color = self.colors.get(class_id, (255, 255, 255))
                
                # 绘制边界框
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 准备标签文本
                    label = f"{class_name} {confidence:.2f}"
                    
                    # 计算文本大小
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # 绘制文本背景
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    
                    # 绘制文本
                    cv2.putText(frame, label, (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 打印检测结果
                print(f"\n检测到物体:")
                print(f"类别: {class_name}")
                print(f"置信度: {confidence:.2%}")
                print(f"位置: ({x1}, {y1}), ({x2}, {y2})")
                print("-" * 30)
        
        return frame

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

def main():
    # 设置GPU
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)

    # 初始化检测器
    detector = YOLODetector(
        model_path='best.pt'  # 使用训练好的最佳模型
    )
    
    # 初始化摄像头
    cap = find_camera()
    if not cap:
        return
    
    # 设置窗口
    if DEBUG_WINDOW:
        window_name = 'YOLOv8检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print(f"- 调试窗口: {'开启' if DEBUG_WINDOW else '关闭'}")
    print("- 按 'q' 键退出程序")
    print("-" * 30)
    
    try:
        while True:
            # 读取摄像头画面
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 进行检测
            frame = detector.detect(frame)
            
            # 显示结果
            if DEBUG_WINDOW:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n程序正常退出")
                    break
            
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
