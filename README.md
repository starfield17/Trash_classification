# Trash_classification

Real-time trash classification system based on YOLO/Fast R-CNN object detection. Utilizes computer vision technology to achieve automatic recognition and classification of trash, and collaborates with automatic sorting equipment through serial communication.

Project Repository: https://github.com/starfield17/Trash_classification.git

## Project Overview

This project implements real-time trash detection and classification through YOLO series models or Fast R-CNN models, integrating the following core functionalities:

- **Real-time Detection**: Uses YOLO or Fast R-CNN object detection algorithms, supports real-time camera feed processing
- **Automatic Classification**: Supports recognition of four major trash categories (kitchen waste, recyclable, hazardous, other)
- **Serial Communication**: Real-time transmission of detection results to STM32 (or other) controllers for automatic sorting
- **Smart Error Prevention**: Built-in anti-duplicate counting and stability detection mechanisms to improve classification accuracy
- **Visual Debugging**: Optional debug window for real-time display of detection results and confidence scores

## Model Comparison

| Feature | YOLO | Fast R-CNN |
|---------|------|------------|
| Speed | Fast | Medium |
| Accuracy | High | Very High |
| Resource Usage | Low | High |
| Use Cases | General hardware and embedded devices | Devices with sufficient computing power (e.g., Jetson) |

# Environment Setup

## Training Environment Setup

### 1. Install System Dependencies
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install basic dependencies
sudo apt install -y build-essential git cmake python3-dev python3-pip wget
```

### 2. Install Conda
```bash
# Download Miniconda / Miniforge
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh #miniconda official source
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh #miniconda Tsinghua source
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh #miniforge git source
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh #miniforge Tsinghua source

# Install
bash ./Miniconda3-latest-Linux-x86_64.sh # miniconda
bash ./Miniforge3-Linux-x86_64.sh # miniforge
# Initialize
source ~/.bashrc
# Verify installation
conda --version
```

### 3. Configure Python Environment
```bash
# Create environment
conda create -n trash_classification python=3.11

# Activate environment
conda activate trash_classification

# Update pip
pip install --upgrade pip
```

### 4. Install CUDA and cuDNN
Please visit NVIDIA official website to download and install corresponding versions:
- CUDA: https://developer.nvidia.com/cuda
- cuDNN: https://developer.nvidia.com/cudnn

### 5. Install Dependencies

#### YOLO Model Dependencies
```bash
# Install PyTorch & other dependencies
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision ultralytics opencv-python numpy scikit-learn
```

#### Fast R-CNN Model Dependencies
```bash
# Install PyTorch & other dependencies
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision tqdm opencv-python numpy scikit-learn concurrent-log-handler
```

### 6. Verify Environment
```bash
# Verify PyTorch GPU support
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Verify YOLO environment
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Verify Fast R-CNN environment
python3 -c "import torchvision; from torchvision.models.detection import fasterrcnn_resnet50_fpn; print('FasterRCNN available')"
```

## Deployment Environment Setup

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### 2. Install Conda and Configure Environment
```bash
# Download Miniconda
#For X86_64 systems, use wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh. Configuration for X86 omitted below, 99% similar to ARM architecture configuration
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh #official source
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh #Tsinghua source
# Install
bash ./Miniconda3-latest-Linux-aarch64.sh
# Initialize
source ~/.bashrc
# Verify installation
conda --version
# Create environment
conda create -n deploy_env python=3.10
conda activate deploy_env
```

### 3. Install Dependencies
```bash
# Basic dependencies
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision opencv-python numpy pyserial transitions

# If using Fast R-CNN model
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ torch torchvision opencv-python numpy pyserial
```

### 4. Configure User Permissions
```bash
# Add user to dialout&video group
sudo usermod -aG dialout $USER
sudo usermod -aG video $USER
# Need to re-login for changes to take effect
# Verify serial port exists
ls -l /dev/ttyAMA* /dev/ttyUSB*
# Verify video device exists
ls -l /dev/video*
```

### 5. Verify Environment
```bash
# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# Test serial port
python3 -c "import serial; print('Serial module ready')"
```

Notes:
- Training environment recommended on GPU servers or high-performance workstations
- Deployment environment can run on regular PCs or Raspberry Pi devices
- Ensure CUDA and PyTorch versions match
- If encountering permission issues, need to re-login or restart system

# Training Guide

## Data Preparation

### Dataset Format
```
label/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### Annotation Format
```json
{
  "labels": [
    {
      "name": "bottle",  // Object class name
      "x1": 100,        // Bounding box top-left x coordinate
      "y1": 100,        // Bounding box top-left y coordinate
      "x2": 200,        // Bounding box bottom-right x coordinate
      "y2": 200         // Bounding box bottom-right y coordinate
    }
  ]
}
```

Supported category mapping:
```python
category_mapping = {
    # Kitchen waste (0)
    'Kitchen_waste': 0,
    'potato': 0,
    'daikon': 0,
    'carrot': 0,
    
    # Recyclable waste (1)
    'Recyclable_waste': 1,
    'bottle': 1,
    'can': 1,
    
    # Hazardous waste (2)
    'Hazardous_waste': 2,
    'battery': 2,
    'drug': 2,
    'inner_packing': 2,
    
    # Other waste (3)
    'Other_waste': 3,
    'tile': 3,
    'stone': 3,
    'brick': 3
}
```

## Training Configuration

### YOLO Model Training

#### Basic Configuration
```python
# train4class_yolovX_easydata.py

# Select base model
select_model = 'yolov12s.pt'  # Options: yolo11s.pt, yolo11m.pt, yolo11l.pt, etc.

# Data path
datapath = './label'  # Point to dataset directory
```

#### Advanced Parameter Tuning
```python
train_args = {
    # Basic training parameters
    'epochs': 120,          # Training epochs
    'batch': 10,           # Batch size
    'imgsz': 640,          # Input image size
    'patience': 15,        # Early stopping epochs
    
    # Optimizer parameters
    'optimizer': 'AdamW',   # Optimizer type
    'lr0': 0.0005,         # Initial learning rate
    'lrf': 0.01,           # Final learning rate ratio
    'momentum': 0.937,     # Momentum parameter
    'weight_decay': 0.0005, # Weight decay
    
    # Warmup parameters
    'warmup_epochs': 10,    # Warmup epochs
    'warmup_momentum': 0.5, # Warmup momentum
    'warmup_bias_lr': 0.05, # Warmup bias learning rate
    
    # Loss weights
    'box': 4.0,            # Bounding box regression loss weight
    'cls': 2.0,            # Classification loss weight
    'dfl': 1.5,            # Distributed focal loss weight
}
```

### Fast R-CNN Model Training

#### Basic Configuration
```python
# FAST_R_CNN_train.py

# Data path
datapath = "./label"  # Point to dataset directory

# Select model type
MODEL_TYPE = "resnet50_fpn"  # Options: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"

# Four-class trash dataset configuration
CLASS_NAMES = ["Kitchen waste", "Recyclable waste", "Hazardous waste", "Other waste"]
```

#### Advanced Parameter Tuning
```python
# Training parameters
num_epochs = min(max(10, len(train_files) // 10), 200)  # Training epochs auto-adjusted based on dataset size
batch_size = 8  # Batch size on GPU
patience = 10  # Early stopping parameter, stop if no improvement for 10 epochs
min_delta = 0.001  # Minimum improvement threshold

# Optimizer parameters
optimizer = torch.optim.SGD(
    params, 
    lr=0.005,  # Initial learning rate 
    momentum=0.9,  # Momentum
    weight_decay=0.0005  # Weight decay
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,     # Reduce by half when plateauing
    patience=3,     # Wait 3 cycles before reducing
    min_lr=1e-6     # Don't go below this value
)
```

## Training Launch

### YOLO Model Training
```bash
python train4class_yolovX_easydata.py
```

### Fast R-CNN Model Training
```bash
python FAST_R_CNN_train.py
```

## Training Process Monitoring

### Output Description
```
Checking dataset integrity...
Found 100 images
Found 98 valid image and label file pairs
Splitting dataset into training, validation and test sets...
Training set: 78 images, Validation set: 10 images, Test set: 10 images
```

### YOLO Training Metrics
- **mAP**: Mean Average Precision
- **P**: Precision
- **R**: Recall
- **loss**: Total loss value
  - box_loss: Bounding box loss
  - cls_loss: Classification loss
  - dfl_loss: Distributed focal loss

### Fast R-CNN Training Metrics
- **train_loss**: Training loss
  - loss_classifier: Classification loss
  - loss_box_reg: Bounding box regression loss
  - loss_objectness: Objectness loss
  - loss_rpn_box_reg: RPN box regression loss
- **val_loss**: Validation loss
  - Contains the same detailed loss components

## Fast R-CNN Model Type Selection

Fast R-CNN model supports multiple different backbone networks:

### 1. resnet50_fpn (Standard)
- Most comprehensive feature extraction capability
- Suitable for scenarios requiring high accuracy
- Higher resource consumption

### 2. resnet18_fpn (Lightweight)
- Good balance between accuracy and speed
- Suitable for regular PCs or edge devices
- Moderate resource consumption

### 3. mobilenet_v3 (Ultra-lightweight)
- Optimized for mobile and embedded systems
- Lowest resource consumption
- Trades some accuracy for speed

### 4. resnet50_fpn_v2 (Improved)
- Improved version based on ResNet50
- Stronger feature extraction capability
- Requires more computational resources

## Common Issues Handling

1. **Out of memory**:
   - Reduce batch_size
   - Lower imgsz or choose lighter models (e.g., resnet18_fpn, mobilenet_v3)
   - Use smaller base models

2. **Overfitting**:
   - Increase weight_decay
   - Reduce epochs
   - Enable data augmentation

3. **Underfitting**:
   - Increase epochs
   - Raise learning rate
   - Increase model capacity or choose stronger models (e.g., resnet50_fpn_v2)

4. **Training instability**:
   - Lower learning rate
   - Increase warmup_epochs
   - Adjust loss weights

# Deployment Guide

## Model Deployment

### YOLO Model Deployment
```bash
python y12e_rebuild.py
```

### Fast R-CNN Model Deployment
```bash
python FAST_R_CNN_deploy.py
```

## Configuration Adjustment

### YOLO Configuration
```python
# Global configuration variables
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9
model_path = "yolov12n_e300.pt"
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
```

### Fast R-CNN Configuration
```python
# Global configuration variables
DEBUG_WINDOW = True
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.7
model_path = "output/model_final.pth"  # Change to FastCNN model path
STM32_PORT = "/dev/ttyUSB0"
STM32_BAUD = 115200
MODEL_TYPE = "resnet50_fpn"  # Model type: "resnet50_fpn", "resnet18_fpn", "mobilenet_v3", "resnet50_fpn_v2"
```

## Common Issues Handling

### Camera Issues
1. **Cannot open camera**:
```bash
# Check device
ls /dev/video*

# Check permissions
sudo usermod -a -G video $USER
```

2. **Frame lag**:
```python
# Reduce resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

### Serial Port Issues
1. **Cannot open serial port**:
```bash
# Check device
ls -l /dev/ttyAMA* /dev/ttyUSB*

# Check permissions
sudo chmod 666 /dev/ttyUSB0
```

2. **Communication instability**:
```python
# Increase timeout
self.port = serial.Serial(
    self.config.stm32_port,
    self.config.stm32_baud,
    timeout=0.2,           # Increase read timeout
    write_timeout=0.2      # Increase write timeout
)
```

### Detection Issues
1. **High false positive rate**:
- Increase confidence threshold (CONF_THRESHOLD)
- Increase stability detection time (min_position_change)
- Adjust camera angle and lighting

2. **High false negative rate**:
- Lower confidence threshold
- Reduce stability requirements
- Improve environmental lighting conditions

3. **Fast R-CNN specific issues**:
- If model loading fails, check if MODEL_TYPE matches training configuration
- For resource-constrained devices, try switching to lightweight models (mobilenet_v3 or resnet18_fpn)

# Troubleshooting Guide

## Performance Optimization

### GPU Usage Optimization
```python
# Check GPU usage
nvidia-smi -l 1  # Real-time GPU monitoring

# Optimize GPU memory
torch.cuda.empty_cache()  # Clear GPU cache
```

### Memory Management
```python
# Clean memory after each main loop iteration
import gc
gc.collect()
```

## Debugging Methods

### Debug Window
Set `DEBUG_WINDOW = True` to enable visual debug window displaying detection results and confidence scores.

### Serial Communication Debugging
When `DEBUG_WINDOW = True`, detailed serial communication packet information will be printed for debugging.

### Error Handling
The code integrates complete error handling and state recovery mechanisms to ensure automatic system recovery under abnormal conditions.

## Fast R-CNN Specific Debugging Tips

### Model Output Analysis
```python
# Analyze first detection result from model
prediction = predictions[0]
print("Bounding boxes:", prediction['boxes'].shape)
print("Scores:", prediction['scores'])
print("Labels:", prediction['labels'])
```

### Performance Comparison of Different Backbones
You can try different backbone networks by modifying the `MODEL_TYPE` parameter and compare their performance on specific datasets.

## Development Recommendations

### Development Workflow
1. **Incremental approach**
   - Start with small datasets for testing
   - Scale up after confirming workflow is correct
   - Gradually enable advanced features

2. **Version control**
   - Save models with different configurations
   - Record experimental results
   - Maintain parameter version management

3. **Testing strategy**
   - Unit test important components
   - Integration test key processes
   - Stress test system stability

## Deployment Optimization

### 1. Model Optimization

#### YOLO Model Optimization
```python
# Model quantization (required for NPU)
from ultralytics.engine.exporter import Exporter
exporter = Exporter()
exporter.export(format='onnx')  # Export to ONNX format
```

#### Fast R-CNN Model Optimization
```python
# Export models in different formats
save_optimized_model(model, output_dir, device, model_type)

# Half precision model
model_fp16 = model.half()
fp16_path = os.path.join(output_dir, "model_fp16.pth")
torch.save(model_fp16.state_dict(), fp16_path)

# ONNX model (suitable for Raspberry Pi deployment)
torch.onnx.export(
    wrapper, 
    dummy_input, 
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=11
)
```

### 2. Inference Optimization
```python
# Use preprocessing batching
# Set non-blocking mode
cv2.setUseOptimized(True)
```

### 3. Memory Optimization
```python
# Regular memory cleanup
import gc

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
```

## Monitoring Metrics

### 1. System Monitoring
```python
def monitor_system():
    import psutil
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    print(f"CPU: {cpu_percent}%, MEM: {mem_percent}%")
```

### 2. Detection Performance Monitoring
Using the statistics manager `StatisticsManager` can record and analyze system detection performance metrics.
