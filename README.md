# Trash_classification
- :(
## 项目简介
本项目利用YOLO目标检测算法，结合OpenCV和PyTorch，实现垃圾的自动识别与分类。系统通过摄像头实时获取图像，检测并分类垃圾类型，并将结果通过串口发送至STM32微控制器，同时在串口屏上显示分类信息。项目支持PyTorch进行模型训练。

## 主要功能
1. **目标检测与分类**：使用YOLO模型识别图像中的垃圾，并分类为厨余垃圾、可回收垃圾、有害垃圾或其他垃圾。
2. **模型训练**：支持PyTorch框架进行模型训练。
3. **串口通信**：将分类结果,坐标通过串口发送至STM32。
4. **调试窗口**：可选的调试窗口显示检测框、类别名称和置信度，便于开发调试。

## 环境配置

### 虚拟环境
推荐使用Conda创建虚拟环境，支持Python 3.10、3.11或3.12。

```bash
conda create -n trash_classification python=3.10
conda activate trash_classification
```

### 依赖安装

- **训练依赖**
  - PyTorch: `pip install torch opencv-python numpy torchvision ultralytics`

- **推理依赖**
  - PyTorch: `pip install torch opencv-python numpy torchvision ultralytics`
