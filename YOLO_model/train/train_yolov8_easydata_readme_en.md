# YOLOv8 Training Pipeline for Object Detection

A comprehensive training pipeline for YOLOv8 object detection model, featuring dataset preparation, validation, augmentation, and optimized training configurations.

## Features

- Automated dataset integrity checking
- Smart train/validation/test split
- Custom data augmentation pipeline
- Optimized YOLOv8 training configurations
- Support for 11 object classes
- Comprehensive error handling

## Prerequisites

### Software Requirements

```bash
pip install ultralytics
pip install opencv-python
pip install scikit-learn
pip install albumentations
pip install pyyaml
```

## Dataset Structure

The expected dataset structure:
```
dataset_directory/
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    ├── image2.json
    └── ...
```

### JSON Label Format
```json
{
    "labels": [
        {
            "name": "class_name",
            "x1": float,
            "y1": float,
            "x2": float,
            "y2": float
        }
    ]
}
```

## Supported Classes

1. potato (0)
2. daikon (1)
3. carrot (2)
4. bottle (3)
5. can (4)
6. battery (5)
7. drug (6)
8. inner_packing (7)
9. tile (8)
10. stone (9)
11. brick (10)

## Usage

1. Prepare your dataset in the required format
2. Configure the dataset path:
```python
datapath = './your_dataset_path'  # Modify this according to your setup
```
3. Run the training script:
```bash
python train_yolov8_easydata.py
```

## Pipeline Stages

### 1. Dataset Validation
- Checks image file integrity
- Validates JSON label format
- Ensures image-label pair consistency
- Filters out corrupted or invalid data

### 2. Dataset Preparation
- Splits data into train/val/test (80/10/10)
- Converts JSON annotations to YOLO format
- Creates organized directory structure
- Generates data.yaml configuration

### 3. Data Augmentation
Implements a carefully tuned augmentation pipeline:
- Brightness and contrast adjustments
- Hue and saturation shifts
- CLAHE enhancement
- Horizontal flipping
- Geometric transformations
- Noise and blur effects

### 4. Training Configuration
Optimized training parameters:
- AdamW optimizer with warm-up
- Custom learning rate scheduling
- Balanced loss weights
- Disabled complex augmentations
- Early stopping with patience
- Regular model checkpointing

## Directory Structure

After running, the following structure is created:
```
./
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── runs/
    └── train/
```

## Training Parameters

Key configuration parameters:
- Epochs: 100
- Image size: 640x640
- Batch size: 16
- Initial learning rate: 0.0005
- Optimizer: AdamW
- Early stopping patience: 50
- Checkpoint save interval: 5 epochs

## Performance Considerations

- GPU acceleration supported
- Multi-worker data loading
- Memory-efficient data handling
- Robust error handling
- Progress monitoring and logging

## Error Handling

The pipeline includes comprehensive error handling for:
- Dataset integrity issues
- File I/O operations
- Training process monitoring
- Resource management
- Invalid data formats

## Contributing

Feel free to submit issues and enhancement requests!

## Author

[Your Name]
