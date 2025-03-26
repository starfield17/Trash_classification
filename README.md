# Trash_classification
- :(

基于 YOLO 目标检测的实时垃圾分类系统。利用计算机视觉技术实现垃圾的自动识别与分类，通过串口通信实现与自动分类设备的协同工作。

## 项目简介

本项目通过 YOLO系列模型实现垃圾的实时检测与分类，集成了以下核心功能：

- **实时检测**：采用 YOLO 目标检测算法，支持摄像头实时画面处理
- **自动分类**：支持四大类垃圾（厨余、可回收、有害、其他）的识别
- **串口通信**：将检测结果实时传输至 STM32 控制器，实现自动分类
- **智能防误**：内置防重复计数和稳定性检测机制，提高分类准确性
- **可视化调试**：可选的调试窗口，实时显示检测结果和置信度

### 技术特点

1. **检测算法**
   - 采用 YOLO 系列模型
   - 支持 GPU 加速推理
   - 可配置的置信度阈值
   - 实时目标定位和分类

2. **防误机制**
   - 1秒稳定性检测期
   - 5秒防重复计数
   - 自动目标跟踪
   - 低置信度过滤

3. **硬件通信**
   - 3字节串口协议
   - 自动坐标映射
   - 实时位置反馈
   - 错误自动恢复

4. **开发特性**
   - 模块化设计
   - 完整调试接口
   - 可配置参数
   - 支持自定义训练
# 环境配置

## 训练环境配置

### 1. 安装系统依赖
```bash
# 更新系统
sudo apt update
sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y build-essential git cmake python3-dev python3-pip wget
```

### 2. 安装Conda
```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh #官方源
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh #清华源
# 安装
bash ./Miniconda3-latest-Linux-x86_64.sh
# 初始化
source ~/.bashrc
# 验证安装
conda --version
```

### 3. 配置Python环境
```bash
# 创建环境
conda create -n trash_classification python=3.11

# 激活环境
conda activate trash_classification

# 更新pip
pip install --upgrade pip
```

### 4. 安装CUDA和cuDNN
请访问NVIDIA官方网站下载并安装对应版本：
- CUDA: https://developer.nvidia.com/cuda
- cuDNN: https://developer.nvidia.com/cudnn

### 5. 安装依赖包
```bash
# 安装PyTorch & 其他依赖
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision ultralytics opencv-python numpy scikit-learn
```

### 6. 验证环境
```bash
# 验证PyTorch GPU支持
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# 验证YOLO环境
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 部署环境配置

### 1. 安装系统依赖
```bash
sudo apt update
sudo apt install -y python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### 2. 安装Conda和配置环境
```bash
# 下载Miniconda
#如果是X86_64系统，使用 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 以下省略对于X86的配置方法，和arm架构配置方法99%相似
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh #官方源
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh #清华源
# 安装
bash ./Miniconda3-latest-Linux-aarch64.sh
# 初始化
source ~/.bashrc
# 验证安装
conda --version
# 创建环境
conda create -n deploy_env python=3.10
conda activate deploy_env
```

### 3. 安装依赖包
```bash
# 基础依赖
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision ultralytics opencv-python numpy pyserial
```

### 4. 配置用户权限
```bash
# 添加用户到 dialout&video 组
sudo usermod -aG dialout $USER
sudo usermod -aG video $USER
# 需重新登录生效
# 验证串口存在
ls -l /dev/ttyAMA* /dev/ttyUSB*
# 验证视频设备存在
ls -l /dev/video*
```

### 5. 验证环境
```bash
# 测试摄像头
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# 测试串口
python3 -c "import serial; print('Serial module ready')"
```

注意事项：
- 训练环境推荐使用GPU服务器或高性能工作站
- 部署环境可以在普通PC或树莓派等设备上运行
- 确保CUDA和PyTorch版本匹配
- 如遇权限问题，需要重新登录或重启系统

# 训练指南

## 数据准备

### 数据集格式
```
label/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### 标注格式
```json
{
  "labels": [
    {
      "name": "bottle",  // 物体类别名称
      "x1": 100,        // 边界框左上角x坐标
      "y1": 100,        // 边界框左上角y坐标
      "x2": 200,        // 边界框右下角x坐标
      "y2": 200         // 边界框右下角y坐标
    }
  ]
}
```

支持的类别映射：
```python
category_mapping = {
    # 厨余垃圾 (0)
    'potato': 0,
    'daikon': 0,
    'carrot': 0,
    
    # 可回收垃圾 (1)
    'bottle': 1,
    'can': 1,
    
    # 有害垃圾 (2)
    'battery': 2,
    'drug': 2,
    'inner_packing': 2,
    
    # 其他垃圾 (3)
    'tile': 3,
    'stone': 3,
    'brick': 3
}
```

## 训练配置

### 基础配置
```python
# train4class_yolovX_easydata.py

# 选择基础模型
select_model = 'yolov12n.pt'  # 可选: yolo11s.pt, yolo11m.pt, yolo11l.pt等

# 数据路径
datapath = './label'  # 指向数据集目录
```

### 高级参数调整
```python
train_args = {
    # 基础训练参数
    'epochs': 120,          # 训练轮数
    'batch': 10,           # 批次大小
    'imgsz': 640,          # 输入图像尺寸
    'patience': 15,        # 早停轮数
    
    # 优化器参数
    'optimizer': 'AdamW',   # 优化器类型
    'lr0': 0.0005,         # 初始学习率
    'lrf': 0.01,           # 最终学习率比例
    'momentum': 0.937,     # 动量参数
    'weight_decay': 0.0005, # 权重衰减
    
    # 预热参数
    'warmup_epochs': 10,    # 预热轮数
    'warmup_momentum': 0.5, # 预热动量
    'warmup_bias_lr': 0.05, # 预热偏置学习率
    
    # 损失权重
    'box': 4.0,            # 边界框回归损失权重
    'cls': 2.0,            # 分类损失权重
    'dfl': 1.5,            # 分布式焦点损失权重
}
```

## 训练启动

1. **基础训练**:
```bash
python train4class_yolovX_easydata.py
```

2. **指定配置模式训练**:
```python
# 在代码中修改配置模式
config = 'focus_accuracy'  # 使用精度优先配置
```

3. **自定义参数训练**:
```python
# 修改train_args中的参数
train_args.update({
    'batch': 16,
    'lr0': 0.001,
    'epochs': 150
})
```

## 训练过程监控

### 输出说明
```
正在检查数据集完整性...
找到 100 张图片
找到 98 对有效的图片和标签文件
Processing train split...
train: 78 images
Processing val split...
val: 10 images
Processing test split...
test: 10 images
```

### 训练指标
- **mAP**: 平均精度均值
- **P**: 精确率
- **R**: 召回率
- **loss**: 总损失值
  - box_loss: 边界框损失
  - cls_loss: 分类损失
  - dfl_loss: 分布式焦点损失

## 常见问题处理

1. **显存不足**:
   - 减小batch_size
   - 降低imgsz
   - 使用更小的基础模型

2. **过拟合**:
   - 增加weight_decay
   - 减少epochs
   - 启用数据增强

3. **欠拟合**:
   - 增加epochs
   - 提高lr0
   - 增大模型容量

4. **训练不稳定**:
   - 降低lr0
   - 增加warmup_epochs
   - 调整损失权重
  # 部署指南

## 常见问题处理

### 摄像头问题
1. **无法打开摄像头**：
```bash
# 检查设备
ls /dev/video*

# 检查权限
sudo usermod -a -G video $USER
```

2. **画面卡顿**：
```python
# 降低分辨率
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

### 串口问题
1. **无法打开串口**：
```bash
# 检查设备
ls -l /dev/ttyAMA*

# 检查权限
sudo chmod 666 /dev/ttyAMA2
```

2. **通信不稳定**：
```python
# 增加超时时间
self.stm32_port = serial.Serial(
    STM32_PORT, 
    STM32_BAUD,
    timeout=0.2,           # 增加读取超时
    write_timeout=0.2      # 增加写入超时
)
```
### 检测问题
1. **误检率高**：
- 提高置信度阈值
- 增加稳定性检测时间
- 调整摄像头角度和光照

2. **漏检率高**：
- 降低置信度阈值
- 减少稳定性要求
- 改善环境光照条件
  
# 故障排除指南

## 性能优化

### GPU使用优化
```python
# 检查GPU使用情况
nvidia-smi -l 1  # 实时监控GPU使用

# 优化GPU内存
torch.cuda.empty_cache()  # 清理GPU缓存
```

### 内存管理
```python
# 使用生成器加载数据
def load_data():
    for item in dataset:
        yield process_item(item)

# 使用迭代器处理
for batch in load_data():
    model(batch)
```

## 调试方法

### 调试日志级别
```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 性能分析
```python
import cProfile
import pstats

# 分析代码性能
profiler = cProfile.Profile()
profiler.enable()
# 运行代码
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
```

## 开发建议

### 开发流程
1. **循序渐进**
   - 先用小数据集测试
   - 确认流程无误后扩大规模
   - 逐步开启高级功能

2. **版本控制**
   - 保存不同配置的模型
   - 记录实验结果
   - 做好参数版本管理

3. **测试策略**
   - 单元测试重要组件
   - 集成测试关键流程
   - 压力测试系统稳定性

## 部署优化

### 1. 模型优化
```python
# 模型量化(npu必须)
from ultralytics.engine.exporter import Exporter
exporter = Exporter()
exporter.export(format='onnx')  # 导出ONNX格式
```

### 2. 推理优化
```python
# 批处理推理
def batch_inference(images, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        results.extend(model(batch))
    return results
```

### 3. 内存优化
```python
# 定期清理内存
import gc

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
```

## 监控指标

### 1. 系统监控
```python
def monitor_system():
    import psutil
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    logging.info(f"CPU: {cpu_percent}%, MEM: {mem_percent}%")
```

