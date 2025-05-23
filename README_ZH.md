# Trash_classification

基于 YOLO/Fast R-CNN 目标检测的实时垃圾分类系统。利用计算机视觉技术实现垃圾的自动识别与分类，通过串口通信实现与自动分类设备的协同工作。

项目仓库: https://github.com/starfield17/Trash_classification.git

## 项目简介

本项目通过 YOLO 系列模型或 Fast R-CNN 模型实现垃圾的实时检测与分类，集成了以下核心功能：

- **实时检测**：采用 YOLO 或 Fast R-CNN 目标检测算法，支持摄像头实时画面处理
- **自动分类**：支持四大类垃圾（厨余、可回收、有害、其他）的识别
- **串口通信**：将检测结果实时传输至STM32(或者其他)控制器，实现自动分类
- **智能防误**：内置防重复计数和稳定性检测机制，提高分类准确性
- **可视化调试**：可选的调试窗口，实时显示检测结果和置信度

## 模型对比

| 特性 | YOLO | Fast R-CNN |
|------|------|------------|
| 速度 | 较快 | 中等 |
| 精度 | 较高 | 高 |
| 资源消耗 | 较低 | 大 |
| 适用场景 | 普通硬件和嵌入式设备 | 具有一定算力的设备(比如jetson)|

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
# 下载Miniconda / Miniforge
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh #miniconda 官方源
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh #miniconda 清华源
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh #miniforge git源
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh #miniforge 清华源

# 安装
bash ./Miniconda3-latest-Linux-x86_64.sh # miniconda
bash ./Miniforge3-Linux-x86_64.sh # miniforge
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

#### YOLO 模型依赖
```bash
# 安装PyTorch & 其他依赖
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision ultralytics opencv-python numpy scikit-learn
```

#### Fast R-CNN 模型依赖
```bash
# 安装PyTorch & 其他依赖
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision tqdm opencv-python numpy scikit-learn concurrent-log-handler
```

### 6. 验证环境
```bash
# 验证PyTorch GPU支持
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# 验证YOLO环境
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 验证Fast R-CNN环境
python3 -c "import torchvision; from torchvision.models.detection import fasterrcnn_resnet50_fpn; print('FasterRCNN available')"
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
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision opencv-python numpy pyserial transitions

# 如果使用Fast R-CNN模型
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision opencv-python numpy pyserial
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
    'Kitchen_waste': 0,
    'potato': 0,
    'daikon': 0,
    'carrot': 0,
    
    # 可回收垃圾 (1)
    'Recyclable_waste': 1,
    'bottle': 1,
    'can': 1,
    
    # 有害垃圾 (2)
    'Hazardous_waste': 2,
    'battery': 2,
    'drug': 2,
    'inner_packing': 2,
    
    # 其他垃圾 (3)
    'Other_waste': 3,
    'tile': 3,
    'stone': 3,
    'brick': 3
}
```

## 训练配置

### YOLO模型训练

#### 基础配置
```python
# train4class_yolovX_easydata.py

# 选择基础模型
select_model = 'yolov12s.pt'  # 可选: yolo11s.pt, yolo11m.pt, yolo11l.pt等

# 数据路径
datapath = './label'  # 指向数据集目录
```

#### 高级参数调整
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

### Fast R-CNN模型训练

#### 基础配置
```python
# FAST_R_CNN_train.py

# 数据路径
datapath = "./label"  # 指向数据集目录

# 选择模型类型
MODEL_TYPE = "resnet50_fpn"  # 可选: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"

# 四分类垃圾数据集配置
CLASS_NAMES = ["厨余垃圾", "可回收垃圾", "有害垃圾", "其他垃圾"]
```

#### 高级参数调整
```python
# 训练参数
num_epochs = min(max(10, len(train_files) // 10), 200)  # 训练轮数根据数据集大小自动调整
batch_size = 8  # GPU上的批次大小
patience = 10  # 早停参数，10个epoch无改进则停止
min_delta = 0.001  # 最小改进阈值

# 优化器参数
optimizer = torch.optim.SGD(
    params, 
    lr=0.005,  # 初始学习率 
    momentum=0.9,  # 动量
    weight_decay=0.0005  # 权重衰减
)

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,     # 平稳时降低一半
    patience=3,     # 等待3个周期再降低
    min_lr=1e-6     # 不低于这个值
)
```

## 训练启动

### YOLO模型训练
```bash
python train4class_yolovX_easydata.py
```

### Fast R-CNN模型训练
```bash
python FAST_R_CNN_train.py
```

## 训练过程监控

### 输出说明
```
正在检查数据集完整性...
找到 100 张图片
找到 98 对有效的图片和标签文件
将数据集划分为训练集、验证集和测试集...
训练集: 78 图片, 验证集: 10 图片, 测试集: 10 图片
```

### YOLO训练指标
- **mAP**: 平均精度均值
- **P**: 精确率
- **R**: 召回率
- **loss**: 总损失值
  - box_loss: 边界框损失
  - cls_loss: 分类损失
  - dfl_loss: 分布式焦点损失

### Fast R-CNN训练指标
- **train_loss**: 训练损失
  - loss_classifier: 分类损失
  - loss_box_reg: 边界框回归损失
  - loss_objectness: 目标性损失
  - loss_rpn_box_reg: RPN框回归损失
- **val_loss**: 验证损失
  - 包含同样的详细损失组成部分

## Fast R-CNN模型类型选择

Fast R-CNN模型支持多种不同的backbone网络:

### 1. resnet50_fpn (标准版)
- 最全面的特征提取能力
- 适合对精度要求高的场景
- 资源消耗较大

### 2. resnet18_fpn (轻量版)
- 精度与速度的良好平衡
- 适合普通PC或边缘设备
- 资源消耗适中

### 3. mobilenet_v3 (超轻量版)
- 专为移动设备和嵌入式系统优化
- 最低的资源消耗
- 牺牲部分精度换取速度

### 4. resnet50_fpn_v2 (改进版)
- 基于ResNet50的改进版本
- 更强的特征提取能力
- 需要更多的计算资源

## 常见问题处理

1. **显存不足**:
   - 减小batch_size
   - 降低imgsz或选择更轻量的模型(如resnet18_fpn, mobilenet_v3)
   - 使用更小的基础模型

2. **过拟合**:
   - 增加weight_decay
   - 减少epochs
   - 启用数据增强

3. **欠拟合**:
   - 增加epochs
   - 提高learning rate
   - 增大模型容量或选择更强的模型(如resnet50_fpn_v2)

4. **训练不稳定**:
   - 降低learning rate
   - 增加warmup_epochs
   - 调整损失权重

# 部署指南

## 模型部署

### YOLO模型部署
```bash
python y12e_rebuild.py
```

### Fast R-CNN模型部署
```bash
python FAST_R_CNN_deploy.py
```

## 配置调整

### YOLO配置
```python
# 全局配置变量
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9
model_path = "yolov12n_e300.pt"
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
```

### Fast R-CNN配置
```python
# 全局配置变量
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.7
model_path = "output/model_final.pth"  # 修改为FastCNN模型路径
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
MODEL_TYPE = "resnet50_fpn"  # 模型类型: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"
```

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
ls -l /dev/ttyAMA* /dev/ttyUSB*

# 检查权限
sudo chmod 666 /dev/ttyUSB0
```

2. **通信不稳定**：
```python
# 增加超时时间
self.port = serial.Serial(
    self.config.stm32_port,
    self.config.stm32_baud,
    timeout=0.2,           # 增加读取超时
    write_timeout=0.2      # 增加写入超时
)
```

### 检测问题
1. **误检率高**：
- 提高置信度阈值 (CONF_THRESHOLD)
- 增加稳定性检测时间 (min_position_change)
- 调整摄像头角度和光照

2. **漏检率高**：
- 降低置信度阈值
- 减少稳定性要求
- 改善环境光照条件

3. **Fast R-CNN特有问题**:
- 如果模型加载失败，检查MODEL_TYPE是否与训练时一致
- 对于资源受限设备，尝试更换为轻量级模型 (mobilenet_v3或resnet18_fpn)

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
# 在每次主循环迭代后清理内存
import gc
gc.collect()
```

## 调试方法

### 调试窗口
设置 `DEBUG_WINDOW = True` 可以启用可视化调试窗口，显示检测结果和置信度。

### 串口通信调试
当 `DEBUG_WINDOW = True` 时，会打印详细的串口通信数据包信息，便于调试。

### 错误捕获
代码中集成了完整的错误捕获和状态恢复机制，确保系统在异常情况下能够自动恢复。

## Fast R-CNN特有的调试技巧

### 模型输出分析
```python
# 分析模型的第一个检测结果
prediction = predictions[0]
print("边界框:", prediction['boxes'].shape)
print("分数:", prediction['scores'])
print("标签:", prediction['labels'])
```

### 不同backbone的性能对比
可以通过修改 `MODEL_TYPE` 参数，尝试不同的backbone网络，对比它们在特定数据集上的性能表现。

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

#### YOLO模型优化
```python
# 模型量化(NPU必须)
from ultralytics.engine.exporter import Exporter
exporter = Exporter()
exporter.export(format='onnx')  # 导出ONNX格式
```

#### Fast R-CNN模型优化
```python
# 导出不同格式的模型
save_optimized_model(model, output_dir, device, model_type)

# 半精度模型
model_fp16 = model.half()
fp16_path = os.path.join(output_dir, "model_fp16.pth")
torch.save(model_fp16.state_dict(), fp16_path)

# ONNX模型(适用于树莓派部署)
torch.onnx.export(
    wrapper, 
    dummy_input, 
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=11
)
```

### 2. 推理优化
```python
# 使用预处理批处理
# 设置非阻塞模式
cv2.setUseOptimized(True)
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
    print(f"CPU: {cpu_percent}%, MEM: {mem_percent}%")
```

### 2. 检测性能监控
使用统计管理器 `StatisticsManager` 可以记录和分析系统的检测性能指标。
