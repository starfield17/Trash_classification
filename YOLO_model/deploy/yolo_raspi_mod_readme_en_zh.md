# YOLOv8目标检测系统文档

## 目录
1. [系统概述](#系统概述)
2. [系统要求](#系统要求)
3. [主要功能](#主要功能)
4. [配置参数](#配置参数)
5. [类和方法说明](#类和方法说明)
6. [通信协议](#通信协议)
7. [错误处理](#错误处理)

## 系统概述
这是一个基于YOLOv8的实时目标检测系统，集成了摄像头采集、目标检测、串口通信等功能。系统可以识别11种不同的物体，并通过串口将检测结果发送给STM32单片机和串口显示屏。

## 系统要求
- Python 3.x
- PyTorch (支持CUDA的版本)
- OpenCV (cv2)
- Ultralytics YOLOv8
- PySerial
- CUDA支持的GPU（可选）

## 主要功能

### 实时目标检测
- 支持11种物体的识别
- 可配置的置信度阈值
- 实时视频流处理

### 多端口串口通信
- STM32通信（GPIO14/15）
- 串口显示屏通信（GPIO0/1）
- 可配置的波特率和通信协议

### 可视化功能
- 调试窗口显示（可选）
- 检测框和标签的实时显示
- 自动颜色分配给不同类别

## 配置参数
```python
DEBUG_WINDOW = False      # 是否显示调试窗口
ENABLE_SERIAL = True      # 是否启用串口通信
CONF_THRESHOLD = 0.5      # 检测置信度阈值

# 串口配置
STM32_PORT = '/dev/ttyS0'     # STM32串口
STM32_BAUD = 9600            # STM32波特率
SCREEN_PORT = '/dev/ttyAMA2'  # 显示屏串口
SCREEN_BAUD = 9600           # 显示屏波特率
```

## 类和方法说明

### SerialManager 类
主要负责串口通信的管理，包括：
- 初始化STM32和显示屏的串口连接
- 管理数据的发送和接收
- 提供串口资源的清理功能

关键方法：
- `__init__()`: 初始化串口连接
- `receive_stm32_data()`: 接收STM32数据的线程函数
- `send_to_screen()`: 发送数据到显示屏
- `send_to_stm32()`: 发送数据到STM32
- `cleanup()`: 清理串口资源

### YOLODetector 类
负责目标检测的核心类，功能包括：
- 加载和管理YOLO模型
- 处理视频帧并执行检测
- 管理检测结果的显示和通信

支持的物体类别：
1. potato（土豆）
2. daikon（萝卜）
3. carrot（胡萝卜）
4. bottle（瓶子）
5. can（罐子）
6. battery（电池）
7. drug（药品）
8. inner_packing（内包装）
9. tile（瓷砖）
10. stone（石头）
11. brick（砖块）

## 通信协议

### STM32通信协议
- 帧头: `\xff\xff`
- 帧尾: `\xff\xff`
- 数据格式: 类别ID（整数）

### 显示屏通信协议
- 结束符: `\xff\xff\xff`
- 编码方式: GB2312
- 命令格式: `t0.txt="文本内容"`

## 错误处理
系统包含以下错误处理机制：
1. GPU检测和回退机制
2. 串口初始化失败处理
3. 摄像头连接错误处理
4. 数据接收异常处理
5. 资源清理机制

---

# YOLOv8 Object Detection System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Main Features](#main-features)
4. [Configuration Parameters](#configuration-parameters)
5. [Classes and Methods](#classes-and-methods)
6. [Communication Protocol](#communication-protocol)
7. [Error Handling](#error-handling)

## System Overview
This is a real-time object detection system based on YOLOv8, integrating camera capture, object detection, and serial communication functionalities. The system can identify 11 different types of objects and send detection results to an STM32 microcontroller and a serial display screen.

## System Requirements
- Python 3.x
- PyTorch (CUDA-enabled version)
- OpenCV (cv2)
- Ultralytics YOLOv8
- PySerial
- CUDA-enabled GPU (optional)

## Main Features

### Real-time Object Detection
- Support for 11 object categories
- Configurable confidence threshold
- Real-time video stream processing

### Multi-port Serial Communication
- STM32 communication (GPIO14/15)
- Serial display communication (GPIO0/1)
- Configurable baud rates and protocols

### Visualization Features
- Debug window display (optional)
- Real-time bounding box and label display
- Automatic color assignment for different classes

## Configuration Parameters
```python
DEBUG_WINDOW = False      # Enable/disable debug window
ENABLE_SERIAL = True      # Enable/disable serial communication
CONF_THRESHOLD = 0.5      # Detection confidence threshold

# Serial Configuration
STM32_PORT = '/dev/ttyS0'     # STM32 serial port
STM32_BAUD = 9600            # STM32 baud rate
SCREEN_PORT = '/dev/ttyAMA2'  # Display screen serial port
SCREEN_BAUD = 9600           # Display screen baud rate
```

## Classes and Methods

### SerialManager Class
Manages serial communication, including:
- Initialization of STM32 and display screen serial connections
- Management of data transmission and reception
- Cleanup of serial resources

Key Methods:
- `__init__()`: Initialize serial connections
- `receive_stm32_data()`: Thread function for receiving STM32 data
- `send_to_screen()`: Send data to display screen
- `send_to_stm32()`: Send data to STM32
- `cleanup()`: Clean up serial resources

### YOLODetector Class
Core class responsible for object detection, including:
- Loading and managing YOLO model
- Processing video frames and performing detection
- Managing detection result display and communication

Supported object categories:
1. potato
2. daikon
3. carrot
4. bottle
5. can
6. battery
7. drug
8. inner_packing
9. tile
10. stone
11. brick

## Communication Protocol

### STM32 Communication Protocol
- Frame Header: `\xff\xff`
- Frame Footer: `\xff\xff`
- Data Format: Class ID (Integer)

### Display Screen Protocol
- End Marker: `\xff\xff\xff`
- Encoding: GB2312
- Command Format: `t0.txt="text content"`

## Error Handling
The system includes the following error handling mechanisms:
1. GPU detection and fallback mechanism
2. Serial port initialization failure handling
3. Camera connection error handling
4. Data reception exception handling
5. Resource cleanup mechanism
