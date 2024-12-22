# Trash_classification
- :(
## 项目简介
本项目利用YOLO目标检测算法，结合OpenCV和PyTorch，实现垃圾的自动识别与分类。系统通过摄像头实时获取图像，检测并分类垃圾类型，并将结果通过串口发送至STM32微控制器，同时在串口屏上显示分类信息。项目支持使用TensorFlow或PyTorch进行模型训练，适应不同需求。

## 主要功能
1. **目标检测与分类**：使用YOLO模型识别图像中的垃圾，并分类为厨余垃圾、可回收垃圾、有害垃圾或其他垃圾。
2. **模型训练**：支持TensorFlow和PyTorch框架进行模型训练，提供灵活的训练环境。
3. **串口通信**：将分类结果通过串口发送至STM32，并在串口屏上显示详细信息。
4. **调试窗口**：可选的调试窗口显示检测框、类别名称和置信度，便于开发调试。

## 环境配置

### 虚拟环境
推荐使用Conda创建虚拟环境，支持Python 3.10、3.11或3.12。TensorFlow与Python 3.10兼容性最佳。

```bash
conda create -n trash_classification python=3.10
conda activate trash_classification
```

### 依赖安装

- **训练依赖**
  - TensorFlow: `pip install tensorflow opencv-python numpy tensorflow-vision`
  - 或 PyTorch: `pip install torch opencv-python numpy torchvision`

- **推理依赖**
  - TensorFlow: `pip install tensorflow opencv-python numpy tensorflow-vision`
  - 或 PyTorch: `pip install torch opencv-python numpy torchvision`

## 使用说明

1. **准备模型**：将训练好的YOLO模型文件（`best.pt`）放置在指定路径。
2. **连接设备**：连接摄像头、STM32微控制器和串口屏，并配置相应的串口参数。
3. **运行程序**：
    ```bash
    conda activate trash_classification
    python your_script.py
    ```
4. **查看结果**：分类结果将通过串口发送至STM32，并在串口屏上显示；调试窗口（若开启）将显示检测框和详细信息。
5. **退出程序**：按`q`键或使用Ctrl+C中断程序，系统将自动释放资源。

## 注意事项
- 确保摄像头正确连接并被系统识别。
- 串口设备路径和波特率需与实际硬件一致。
- 训练和推理时建议使用GPU，以提升速度。
- TensorFlow建议使用Python 3.10，避免版本兼容性问题。

# Project Documentation

## Project Overview
This project leverages the YOLO (You Only Look Once) object detection algorithm, combined with OpenCV and PyTorch, to achieve automatic waste classification. The system captures real-time images from a camera, detects and classifies waste items, sends classification results to an STM32 microcontroller via serial communication, and displays the information on a serial screen. The project supports training models using either TensorFlow or PyTorch frameworks to accommodate different development needs.

## Main Features
1. **Object Detection and Classification**: Utilizes the YOLO model to identify waste items in images and classify them into categories such as kitchen waste, recyclable waste, hazardous waste, or other waste.
2. **Model Training**: Supports training using TensorFlow or PyTorch, providing flexible environments for different frameworks.
3. **Serial Communication**: Sends classification results to an STM32 microcontroller and displays detailed information on a serial screen.
4. **Debug Window**: An optional debug window displays bounding boxes, class names, and confidence scores to aid in development and troubleshooting.

## Environment Setup

### Virtual Environment
It is recommended to use Conda to create a virtual environment, supporting Python versions 3.10, 3.11, or 3.12. TensorFlow has the best compatibility with Python 3.10.

```bash
conda create -n trash_classification python=3.10
conda activate trash_classification
```

### Dependencies

- **Training Dependencies**
  - TensorFlow: `pip install tensorflow opencv-python numpy tensorflow-vision`
  - Or PyTorch: `pip install torch opencv-python numpy torchvision`

- **Inference Dependencies**
  - TensorFlow: `pip install tensorflow opencv-python numpy tensorflow-vision`
  - Or PyTorch: `pip install torch opencv-python numpy torchvision`

## Usage Instructions

1. **Prepare the Model**: Place the trained YOLO model file (`best.pt`) in the specified path.
2. **Connect Devices**: Connect the camera, STM32 microcontroller, and serial screen, and configure the corresponding serial parameters.
3. **Run the Program**:
    ```bash
    conda activate trash_classification
    python your_script.py
    ```
4. **View Results**: Classification results will be sent to the STM32 and displayed on the serial screen. If the debug window is enabled, detection boxes and detailed information will be visible.
5. **Exit the Program**: Press the `q` key or use Ctrl+C to terminate the program. The system will automatically release resources.

## Notes
- Ensure the camera is properly connected and recognized by the system.
- Serial device paths and baud rates should match the actual hardware configurations.
- Use a GPU for training and inference to enhance performance.
- TensorFlow is best used with Python 3.10 to avoid compatibility issues.
