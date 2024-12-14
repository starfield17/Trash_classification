# 项目文档

## 中文说明

### 项目简介
本项目基于YOLO（You Only Look Once）目标检测算法，结合OpenCV和PyTorch，实现了垃圾分类的自动识别与分类。系统通过摄像头获取实时图像，检测图中的垃圾类别，并将分类结果通过串口发送至STM32微控制器，同时在串口屏上显示分类信息。

### 主要功能
1. **目标检测与分类**：使用YOLO模型检测图像中的垃圾物体，并根据预设的类别映射将其分类为厨余垃圾、可回收垃圾、有害垃圾或其他垃圾。
2. **串口通信**：将检测到的分类结果通过串口发送至STM32微控制器，并在串口屏上显示具体的分类信息。
3. **调试窗口**：可选的调试窗口显示检测结果，包括边界框、类别名称和置信度等信息，便于开发和调试。

### 依赖库
- `cv2` (OpenCV)：用于图像处理和摄像头视频流捕捉。
- `torch`：PyTorch库，用于加载和运行YOLO模型。
- `serial`：用于串口通信。
- `ultralytics`：YOLO模型的实现库。
- `numpy`：数值计算库。
- `threading`、`time`、`subprocess`、`sys`：标准Python库，用于多线程、时间管理等功能。

### 代码结构

#### 全局变量
- `DEBUG_WINDOW`：控制是否开启调试窗口。
- `ENABLE_SERIAL`：控制是否启用串口通信。
- `CONF_THRESHOLD`：检测的置信度阈值。
- 串口配置参数，如`STM32_PORT`、`STM32_BAUD`、`SCREEN_PORT`、`SCREEN_BAUD`等。
- 串口通信协议帧头和帧尾定义。

#### GPU设置
`setup_gpu`函数用于检查是否有可用的GPU，并返回设备信息。如果未检测到GPU，将使用CPU进行推理。

#### `SerialManager`类
负责管理与STM32和串口屏的串口通信，包括初始化串口、发送数据和接收数据。
- **初始化**：打开STM32和串口屏的串口连接，并启动接收STM32数据的线程。
- **`receive_stm32_data`方法**：持续读取STM32发送的数据并打印。
- **`send_to_screen`方法**：向串口屏发送分类信息，控制发送间隔并处理缓冲区。
- **`send_to_stm32`方法**：向STM32发送分类ID，控制发送间隔并处理缓冲区。
- **`cleanup`方法**：关闭串口连接，释放资源。

#### `WasteClassifier`类
负责将细分类别映射到大分类，并提供分类信息的打印功能。
- **类别映射**：将具体物品类别映射到大分类（如厨余垃圾、可回收垃圾等）。
- **`get_category_info`方法**：获取给定类别ID的详细分类信息。
- **`print_classification`方法**：打印分类信息，并返回显示文本。

#### `YOLODetector`类
负责加载YOLO模型并进行目标检测。
- **初始化**：加载YOLO模型，设置类别名称和颜色映射，初始化`SerialManager`。
- **`detect`方法**：对输入帧进行检测，处理最高置信度的检测结果，绘制检测框（如果开启调试窗口），并通过串口发送分类结果。

#### 辅助函数
- **`find_camera`函数**：查找可用的摄像头，返回摄像头对象。
- **`main`函数**：程序的主入口，初始化设备、加载模型、启动摄像头，并循环读取帧进行检测和显示。

### 使用说明
1. **环境配置**：确保已安装所有依赖库，并将YOLO模型文件（`best.pt`）放置在指定路径。
2. **连接设备**：连接摄像头、STM32微控制器和串口屏，并配置相应的串口参数。
3. **运行程序**：执行脚本，系统将启动摄像头，进行实时垃圾分类检测。
4. **查看结果**：分类结果将通过串口发送至STM32，并在串口屏上显示；如果开启调试窗口，还可在窗口中查看检测框和详细信息。
5. **退出程序**：按下`q`键或通过键盘中断（Ctrl+C）退出程序，系统将自动释放资源。

### 注意事项
- 确保摄像头正确连接并能被系统识别。
- 串口设备路径和波特率配置应与实际硬件一致。
- 若无GPU，系统将自动切换至CPU进行推理，可能影响检测速度。

---

## Project Documentation

### Project Overview
This project leverages the YOLO (You Only Look Once) object detection algorithm, combined with OpenCV and PyTorch, to achieve automatic waste classification. The system captures real-time images from a camera, detects and classifies waste items, sends classification results to an STM32 microcontroller via serial communication, and displays the information on a serial screen.

### Main Features
1. **Object Detection and Classification**: Utilizes the YOLO model to detect waste items in images and classifies them into categories such as kitchen waste, recyclable waste, hazardous waste, or other waste based on predefined mappings.
2. **Serial Communication**: Sends detected classification results to an STM32 microcontroller and displays detailed classification information on a serial screen.
3. **Debug Window**: An optional debug window displays detection results, including bounding boxes, class names, and confidence scores, facilitating development and troubleshooting.

### Dependencies
- `cv2` (OpenCV): For image processing and capturing video streams from the camera.
- `torch`: PyTorch library for loading and running the YOLO model.
- `serial`: For serial communication.
- `ultralytics`: YOLO model implementation library.
- `numpy`: Numerical computing library.
- `threading`, `time`, `subprocess`, `sys`: Standard Python libraries for multi-threading, time management, and other functionalities.

### Code Structure

#### Global Variables
- `DEBUG_WINDOW`: Controls whether the debug window is enabled.
- `ENABLE_SERIAL`: Controls whether serial communication is enabled.
- `CONF_THRESHOLD`: Confidence threshold for detections.
- Serial configuration parameters such as `STM32_PORT`, `STM32_BAUD`, `SCREEN_PORT`, `SCREEN_BAUD`, etc.
- Definitions for frame headers and footers for serial communication protocols.

#### GPU Setup
The `setup_gpu` function checks for the availability of a GPU and returns device information. If no GPU is detected, the system defaults to using the CPU for inference.

#### `SerialManager` Class
Manages serial communication with the STM32 microcontroller and the serial screen, including initializing serial ports, sending data, and receiving data.
- **Initialization**: Opens serial connections for STM32 and the serial screen and starts a thread to receive data from STM32.
- **`receive_stm32_data` Method**: Continuously reads data from STM32 and prints it.
- **`send_to_screen` Method**: Sends classification information to the serial screen, controlling send intervals and handling buffers.
- **`send_to_stm32` Method**: Sends classification IDs to STM32, controlling send intervals and handling buffers.
- **`cleanup` Method**: Closes serial connections and releases resources.

#### `WasteClassifier` Class
Handles mapping specific waste categories to broader classifications and provides functionality to print classification information.
- **Category Mapping**: Maps specific item categories to broader categories such as kitchen waste, recyclable waste, hazardous waste, and other waste.
- **`get_category_info` Method**: Retrieves detailed classification information for a given category ID.
- **`print_classification` Method**: Prints classification information and returns display text.

#### `YOLODetector` Class
Responsible for loading the YOLO model and performing object detection.
- **Initialization**: Loads the YOLO model, sets up class names and color mappings, and initializes `SerialManager`.
- **`detect` Method**: Processes input frames, handles the highest confidence detection, draws bounding boxes (if debug window is enabled), and sends classification results via serial communication.

#### Helper Functions
- **`find_camera` Function**: Searches for available cameras and returns a camera object.
- **`main` Function**: The main entry point of the program. Initializes devices, loads the model, starts the camera, and enters a loop to read frames, perform detection, and display results.

### Usage Instructions
1. **Environment Setup**: Ensure all dependencies are installed and place the YOLO model file (`best.pt`) in the specified path.
2. **Connect Devices**: Connect the camera, STM32 microcontroller, and serial screen, configuring the appropriate serial parameters.
3. **Run the Program**: Execute the script. The system will initialize the camera and start real-time waste classification detection.
4. **View Results**: Classification results will be sent to the STM32 and displayed on the serial screen. If the debug window is enabled, detection boxes and detailed information will be visible in the window.
5. **Exit the Program**: Press the `q` key or use a keyboard interrupt (Ctrl+C) to exit the program. The system will automatically release resources.

### Notes
- Ensure the camera is correctly connected and recognized by the system.
- Serial device paths and baud rates should match the actual hardware configurations.
- If no GPU is available, the system will switch to CPU inference, which may impact detection speed.
