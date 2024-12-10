# YOLOv8 Object Detection System for Raspberry Pi

This project implements a real-time object detection system using YOLOv8 on Raspberry Pi, with integrated serial communication capabilities for STM32 and a serial display screen.

## Features

- Real-time object detection using YOLOv8
- GPU acceleration support (when available)
- Dual serial communication:
  - STM32 microcontroller interface
  - Serial display screen output
- Live video feed display (optional debug window)
- Support for 11 object classes
- Configurable confidence threshold
- Automatic camera detection

## Prerequisites

### Hardware Requirements

- Raspberry Pi (with camera module)
- STM32 microcontroller (connected via UART)
- Serial display screen
- GPU (optional, for acceleration)

### Software Dependencies

```bash
pip install torch
pip install ultralytics
pip install opencv-python
pip install pyserial
```

## Configuration

### Serial Port Settings

- STM32 Communication:
  - Port: `/dev/ttyS0`
  - Baud Rate: 9600
  - Protocol: Custom frame format with headers and footers

- Display Screen:
  - Port: `/dev/ttyUSB0`
  - Baud Rate: 115200
  - Encoding: GB2312

### Global Parameters

```python
DEBUG_WINDOW = True     # Enable/disable debug window
ENABLE_SERIAL = True    # Enable/disable serial communication
CONF_THRESHOLD = 0.5    # Detection confidence threshold
```

## Object Classes

The system can detect the following objects:

1. Potato (0)
2. Daikon (1)
3. Carrot (2)
4. Bottle (3)
5. Can (4)
6. Battery (5)
7. Drug (6)
8. Inner Packing (7)
9. Tile (8)
10. Stone (9)
11. Brick (10)

## Communication Protocol

### STM32 Protocol

- Frame Format: `[FF FF] [Data] [FF FF]`
- Data: Single byte representing the detected class ID
- Bidirectional communication supported

### Display Screen Protocol

- Command Format: `t0.txt="[Text]"[FF FF FF]`
- Text Encoding: GB2312
- One-way communication (output only)

## Usage

1. Connect your hardware components (camera, STM32, display screen)
2. Verify the serial port configurations match your setup
3. Place your trained YOLOv8 model file (`best.pt`) in the project directory
4. Run the program:

```bash
python yolo_raspi_mod.py
```

### Runtime Controls

- Press 'q' to quit the program when debug window is enabled
- Use Ctrl+C to terminate the program from terminal

## Program Flow

1. System initialization:
   - GPU detection and setup
   - Serial port initialization
   - Camera detection
   - Model loading

2. Main loop:
   - Frame capture from camera
   - Object detection using YOLOv8
   - Results visualization (if debug enabled)
   - Serial communication of results
   - Display screen updates

## Error Handling

The system includes comprehensive error handling for:
- Camera initialization failures
- Serial communication errors
- GPU availability checks
- Frame capture issues
- Model loading problems

## Debug Features

When `DEBUG_WINDOW` is enabled:
- Live video feed with detection boxes
- Confidence scores display
- Color-coded object classes
- Resizable window (800x600 default)

## Performance Considerations

- GPU acceleration is automatically utilized when available
- Threading is used for serial communication to prevent I/O blocking
- Buffer management for serial communication
- Configurable confidence threshold to filter detections

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PySerial developers
