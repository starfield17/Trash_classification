"""
Configuration module for YOLO training application.
Contains all configurable global variables and training profiles.
"""

import os

# Model selection
SELECT_MODEL = "yolo12s.pt"

# Default data path
DATAPATH = "./label"

# Default training configuration
DEFAULT_TRAIN_CONFIG = {
    "epochs": 120,
    "imgsz": 640,
    "batch": 10,
    "patience": 15,
    "save_period": 5,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 10,
    "warmup_momentum": 0.5,
    "warmup_bias_lr": 0.05,
    "box": 4.0,
    "cls": 2.0,
    "dfl": 1.5,
    "close_mosaic": 0,
    "nbs": 64,
    "overlap_mask": False,
    "multi_scale": True,
    "single_cls": False,
    "rect": True,
    "cache": True,
    "exist_ok": True,
}

# Training profiles for different scenarios
TRAINING_PROFILES = {
    "default": {
        # Uses DEFAULT_TRAIN_CONFIG as is
    },
    "large_dataset": {
        "batch": 32,
        "lr0": 0.001,
        "epochs": 150,
        "patience": 30,
    },
    "small_dataset": {
        "batch": 16,
        "lr0": 0.0001,
        "weight_decay": 0.001,
        "warmup_epochs": 15,
    },
    "focus_accuracy": {
        "imgsz": 640,
        "box": 7.5,
        "cls": 4.0,
        "dfl": 3.0,
        "patience": 20,
        "batch": 16,
        "epochs": 300,
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "dropout": 0.1,
    },
    "focus_speed": {
        "imgsz": 512,
        "epochs": 150,
        "patience": 30,
        "batch": 48,
    },
    "servermode": {
        "box": 7.5,
        "cls": 4.0,
        "dfl": 3.0,
        "patience": 15,
        "epochs": 150,
        "dropout": 0.1,
        "imgsz": 640,
        "batch": 32,
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "workers": max(1, int(os.cpu_count() / 2) if os.cpu_count() else 1),
        "device": "0",
        "half": True,
        "cache": "ram",
        "cos_lr": True,
        "warmup_epochs": 10,
        "overlap_mask": True,
        "save_period": 5,
        "multi_scale": True,
    }
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    "enabled": {
        "augment": True,
        "degrees": 10.0,
        "scale": 0.2,
        "fliplr": 0.2,
        "flipud": 0.2,
    },
    "disabled": {
        "augment": False,
        "degrees": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0,
        "mixup": 0,
        "copy_paste": 0,
    }
}

# Category mapping for four major waste categories
CATEGORY_MAPPING = {
    # Kitchen Waste (0)
    "Kitchen_waste": 0,
    "potato": 0,
    "daikon": 0,
    "carrot": 0,
    # Recyclable Waste (1)
    "Recyclable_waste": 1,
    "bottle": 1,
    "can": 1,
    # Hazardous Waste (2)
    "Hazardous_waste": 2,
    "battery": 2,
    "drug": 2,
    "inner_packing": 2,
    # Other Waste (3)
    "Other_waste": 3,
    "tile": 3,
    "stone": 3,
    "brick": 3,
}

# Data configuration for YAML
DATA_CONFIG = {
    "names": {
        0: "Kitchen Waste",
        1: "Recyclable Waste",
        2: "Hazardous Waste",
        3: "Other Waste"
    },
    "nc": 4,  # Number of classes
}

# Image file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
