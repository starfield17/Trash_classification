#!/usr/bin/env python3
"""
YOLO Training GUI Interface
User-friendly graphical interface for training YOLO models using PyQt6.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from datetime import datetime
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTextEdit, QFileDialog, QMessageBox,
    QTabWidget, QProgressBar, QSlider, QGridLayout, QScrollArea,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QTextCursor

from train_core import TrainingConfig, TrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingThread(QThread):
    """Thread for running training without blocking the GUI"""
    progress_update = pyqtSignal(str, int)  # message, percentage
    log_update = pyqtSignal(str)  # log message
    finished_signal = pyqtSignal(dict)  # results
    error_signal = pyqtSignal(str)  # error message
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.pipeline = TrainingPipeline(config)
        self._is_running = True
    
    def progress_callback(self, message: str, progress: int):
        """Callback for progress updates"""
        if self._is_running:
            self.progress_update.emit(message, progress)
            self.log_update.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def run(self):
        """Run training in thread"""
        try:
            self.log_update.emit(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results = self.pipeline.run(progress_callback=self.progress_callback)
            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))
            logger.error(f"Training error: {e}")
    
    def stop(self):
        """Stop the training thread"""
        self._is_running = False


class ConfigWidget(QWidget):
    """Widget for configuration parameters"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the configuration UI"""
        layout = QVBoxLayout()
        
        # Create tabs for different configuration sections
        self.tabs = QTabWidget()
        
        # Basic Settings Tab
        self.basic_tab = self.create_basic_tab()
        self.tabs.addTab(self.basic_tab, "Basic Settings")
        
        # Training Parameters Tab
        self.training_tab = self.create_training_tab()
        self.tabs.addTab(self.training_tab, "Training Parameters")
        
        # Optimization Tab
        self.optimization_tab = self.create_optimization_tab()
        self.tabs.addTab(self.optimization_tab, "Optimization")
        
        # Augmentation Tab
        self.augmentation_tab = self.create_augmentation_tab()
        self.tabs.addTab(self.augmentation_tab, "Augmentation")
        
        # Advanced Tab
        self.advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    def browse_model(self):
        """Browse for a model file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt)"
        )
        if path:
            self.model_combo.setCurrentText(path)
    def create_basic_tab(self) -> QWidget:
        """Create basic settings tab"""
        widget = QWidget()
        layout = QGridLayout()
                
        # Model selection
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)  # Make the combo box editable
        self.model_combo.addItems([
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
            "yolo12n.pt",
            "yolo12s.pt",
            "yolo12m.pt",
            "yolo12l.pt",
            "yolo12x.pt"
        ])

        self.model_combo.setCurrentText("yolov8s.pt")
        self.model_combo.lineEdit().setPlaceholderText("Enter model name or path...")
        layout.addWidget(self.model_combo, 0, 1)

        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        layout.addWidget(self.browse_model_btn, 0, 2)
        
        # Preset configuration
        layout.addWidget(QLabel("Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "default", "large_dataset", "small_dataset",
            "focus_accuracy", "focus_speed", "servermode"
        ])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        layout.addWidget(self.preset_combo, 1, 1)
        
        # Dataset path
        layout.addWidget(QLabel("Dataset Path:"), 2, 0)
        self.data_path_edit = QLineEdit("./label")
        layout.addWidget(self.data_path_edit, 2, 1)
        
        self.browse_data_btn = QPushButton("Browse...")
        self.browse_data_btn.clicked.connect(self.browse_dataset)
        layout.addWidget(self.browse_data_btn, 2, 2)
        
        # Output directory
        layout.addWidget(QLabel("Output Directory:"), 3, 0)
        self.output_dir_edit = QLineEdit("")
        layout.addWidget(self.output_dir_edit, 3, 1)
        
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        layout.addWidget(self.browse_output_btn, 3, 2)
        
        # Run name
        layout.addWidget(QLabel("Run Name:"), 4, 0)
        self.run_name_edit = QLineEdit("runs/train")
        layout.addWidget(self.run_name_edit, 4, 1)
        
        # Dataset splits
        split_group = QGroupBox("Dataset Splits")
        split_layout = QGridLayout()
        
        split_layout.addWidget(QLabel("Train:"), 0, 0)
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.1, 0.9)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.9)
        split_layout.addWidget(self.train_split_spin, 0, 1)
        
        split_layout.addWidget(QLabel("Validation:"), 1, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.3)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.05)
        split_layout.addWidget(self.val_split_spin, 1, 1)
        
        split_layout.addWidget(QLabel("Test:"), 2, 0)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.05, 0.3)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.05)
        split_layout.addWidget(self.test_split_spin, 2, 1)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group, 5, 0, 1, 3)
        
        # Options
        self.use_symlinks_check = QCheckBox("Use symbolic links (saves disk space)")
        self.use_symlinks_check.setChecked(True)
        layout.addWidget(self.use_symlinks_check, 6, 0, 1, 2)
        
        self.resume_check = QCheckBox("Resume from last checkpoint")
        layout.addWidget(self.resume_check, 7, 0, 1, 2)
        
        # Add stretch to push everything to the top
        layout.setRowStretch(8, 1)
        
        widget.setLayout(layout)
        return widget
    
    def create_training_tab(self) -> QWidget:
        """Create training parameters tab"""
        widget = QWidget()
        layout = QGridLayout()
        
        # Epochs
        layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(120)
        layout.addWidget(self.epochs_spin, 0, 1)
        
        # Batch size
        layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(10)
        layout.addWidget(self.batch_size_spin, 1, 1)
        
        # Image size
        layout.addWidget(QLabel("Image Size:"), 2, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "768", "1024"])
        self.imgsz_combo.setCurrentText("640")
        layout.addWidget(self.imgsz_combo, 2, 1)
        
        # Device
        layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "1", "2", "3"])
        self.device_combo.setCurrentText("auto")
        layout.addWidget(self.device_combo, 3, 1)
        
        # Workers
        layout.addWidget(QLabel("Workers:"), 4, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(-1, 32)
        self.workers_spin.setValue(-1)
        self.workers_spin.setSpecialValueText("Auto")
        layout.addWidget(self.workers_spin, 4, 1)
        
        # Patience
        layout.addWidget(QLabel("Early Stop Patience:"), 5, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(15)
        layout.addWidget(self.patience_spin, 5, 1)
        
        # Save period
        layout.addWidget(QLabel("Save Period:"), 6, 0)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(1, 50)
        self.save_period_spin.setValue(5)
        layout.addWidget(self.save_period_spin, 6, 1)
        
        # Options
        self.exist_ok_check = QCheckBox("Overwrite existing results")
        self.exist_ok_check.setChecked(True)
        layout.addWidget(self.exist_ok_check, 7, 0, 1, 2)
        
        layout.setRowStretch(8, 1)
        widget.setLayout(layout)
        return widget
    
    def create_optimization_tab(self) -> QWidget:
        """Create optimization parameters tab"""
        widget = QWidget()
        layout = QGridLayout()
        
        # Optimizer
        layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "Adam", "SGD", "RMSProp"])
        layout.addWidget(self.optimizer_combo, 0, 1)
        
        # Learning rate
        layout.addWidget(QLabel("Initial LR:"), 1, 0)
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(6)
        self.lr0_spin.setRange(0.000001, 1.0)
        self.lr0_spin.setSingleStep(0.0001)
        self.lr0_spin.setValue(0.0005)
        layout.addWidget(self.lr0_spin, 1, 1)
        
        # Final LR ratio
        layout.addWidget(QLabel("Final LR Ratio:"), 2, 0)
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setDecimals(4)
        self.lrf_spin.setRange(0.0001, 1.0)
        self.lrf_spin.setSingleStep(0.01)
        self.lrf_spin.setValue(0.01)
        layout.addWidget(self.lrf_spin, 2, 1)
        
        # Momentum
        layout.addWidget(QLabel("Momentum:"), 3, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setDecimals(4)
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setValue(0.937)
        layout.addWidget(self.momentum_spin, 3, 1)
        
        # Weight decay
        layout.addWidget(QLabel("Weight Decay:"), 4, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        layout.addWidget(self.weight_decay_spin, 4, 1)
        
        # Warmup epochs
        layout.addWidget(QLabel("Warmup Epochs:"), 5, 0)
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 50)
        self.warmup_epochs_spin.setValue(10)
        layout.addWidget(self.warmup_epochs_spin, 5, 1)
        
        # Loss weights group
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QGridLayout()
        
        loss_layout.addWidget(QLabel("Box:"), 0, 0)
        self.box_spin = QDoubleSpinBox()
        self.box_spin.setRange(0.1, 10.0)
        self.box_spin.setSingleStep(0.5)
        self.box_spin.setValue(4.0)
        loss_layout.addWidget(self.box_spin, 0, 1)
        
        loss_layout.addWidget(QLabel("Classification:"), 1, 0)
        self.cls_spin = QDoubleSpinBox()
        self.cls_spin.setRange(0.1, 10.0)
        self.cls_spin.setSingleStep(0.5)
        self.cls_spin.setValue(2.0)
        loss_layout.addWidget(self.cls_spin, 1, 1)
        
        loss_layout.addWidget(QLabel("DFL:"), 2, 0)
        self.dfl_spin = QDoubleSpinBox()
        self.dfl_spin.setRange(0.1, 10.0)
        self.dfl_spin.setSingleStep(0.5)
        self.dfl_spin.setValue(1.5)
        loss_layout.addWidget(self.dfl_spin, 2, 1)
        
        loss_group.setLayout(loss_layout)
        layout.addWidget(loss_group, 6, 0, 1, 2)
        
        layout.setRowStretch(7, 1)
        widget.setLayout(layout)
        return widget
    
    def create_augmentation_tab(self) -> QWidget:
        """Create augmentation settings tab"""
        widget = QWidget()
        layout = QGridLayout()
        
        # Enable augmentation
        self.augment_check = QCheckBox("Enable Data Augmentation")
        self.augment_check.stateChanged.connect(self.on_augment_changed)
        layout.addWidget(self.augment_check, 0, 0, 1, 2)
        
        # Augmentation parameters
        layout.addWidget(QLabel("Rotation Degrees:"), 1, 0)
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setValue(10.0)
        self.degrees_spin.setEnabled(False)
        layout.addWidget(self.degrees_spin, 1, 1)
        
        layout.addWidget(QLabel("Scale Range:"), 2, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(0.2)
        self.scale_spin.setEnabled(False)
        layout.addWidget(self.scale_spin, 2, 1)
        
        layout.addWidget(QLabel("Flip LR Probability:"), 3, 0)
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.1)
        self.fliplr_spin.setValue(0.2)
        self.fliplr_spin.setEnabled(False)
        layout.addWidget(self.fliplr_spin, 3, 1)
        
        layout.addWidget(QLabel("Flip UD Probability:"), 4, 0)
        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setSingleStep(0.1)
        self.flipud_spin.setValue(0.2)
        self.flipud_spin.setEnabled(False)
        layout.addWidget(self.flipud_spin, 4, 1)
        
        layout.setRowStretch(5, 1)
        widget.setLayout(layout)
        return widget
    
    def create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QGridLayout()
        
        # Advanced options
        self.mixed_precision_check = QCheckBox("Mixed Precision Training (FP16)")
        layout.addWidget(self.mixed_precision_check, 0, 0, 1, 2)
        
        self.multi_scale_check = QCheckBox("Multi-Scale Training")
        self.multi_scale_check.setChecked(True)
        layout.addWidget(self.multi_scale_check, 1, 0, 1, 2)
        
        self.rect_check = QCheckBox("Rectangular Training")
        self.rect_check.setChecked(True)
        layout.addWidget(self.rect_check, 2, 0, 1, 2)
        
        self.cache_check = QCheckBox("Cache Images in Memory")
        self.cache_check.setChecked(True)
        layout.addWidget(self.cache_check, 3, 0, 1, 2)
        
        self.overlap_mask_check = QCheckBox("Overlap Mask")
        layout.addWidget(self.overlap_mask_check, 4, 0, 1, 2)
        
        self.single_cls_check = QCheckBox("Single Class Mode")
        layout.addWidget(self.single_cls_check, 5, 0, 1, 2)
        
        self.cos_lr_check = QCheckBox("Cosine LR Scheduler")
        layout.addWidget(self.cos_lr_check, 6, 0, 1, 2)
        
        # Dropout
        layout.addWidget(QLabel("Dropout:"), 7, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.0)
        layout.addWidget(self.dropout_spin, 7, 1)
        
        # Close mosaic
        layout.addWidget(QLabel("Close Mosaic:"), 8, 0)
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(0)
        layout.addWidget(self.close_mosaic_spin, 8, 1)
        
        layout.setRowStretch(9, 1)
        widget.setLayout(layout)
        return widget
    
    def on_preset_changed(self, preset: str):
        """Handle preset change"""
        if preset == "large_dataset":
            self.batch_size_spin.setValue(32)
            self.lr0_spin.setValue(0.001)
            self.epochs_spin.setValue(150)
            self.patience_spin.setValue(30)
        elif preset == "small_dataset":
            self.batch_size_spin.setValue(16)
            self.lr0_spin.setValue(0.0001)
            self.weight_decay_spin.setValue(0.001)
            self.warmup_epochs_spin.setValue(15)
        elif preset == "focus_accuracy":
            self.imgsz_combo.setCurrentText("640")
            self.box_spin.setValue(7.5)
            self.cls_spin.setValue(4.0)
            self.dfl_spin.setValue(3.0)
            self.patience_spin.setValue(20)
            self.batch_size_spin.setValue(16)
            self.epochs_spin.setValue(300)
            self.lr0_spin.setValue(0.001)
            self.dropout_spin.setValue(0.1)
        elif preset == "focus_speed":
            self.imgsz_combo.setCurrentText("512")
            self.epochs_spin.setValue(150)
            self.patience_spin.setValue(30)
            self.batch_size_spin.setValue(48)
        elif preset == "servermode":
            self.box_spin.setValue(7.5)
            self.cls_spin.setValue(4.0)
            self.dfl_spin.setValue(3.0)
            self.patience_spin.setValue(15)
            self.epochs_spin.setValue(150)
            self.dropout_spin.setValue(0.1)
            self.batch_size_spin.setValue(32)
            self.mixed_precision_check.setChecked(True)
            self.cos_lr_check.setChecked(True)
            self.overlap_mask_check.setChecked(True)
    
    def on_augment_changed(self, state):
        """Handle augmentation checkbox change"""
        enabled = state == Qt.CheckState.Checked
        self.degrees_spin.setEnabled(enabled)
        self.scale_spin.setEnabled(enabled)
        self.fliplr_spin.setEnabled(enabled)
        self.flipud_spin.setEnabled(enabled)
    
    def browse_dataset(self):
        """Browse for dataset directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self.data_path_edit.setText(path)
    
    def browse_output(self):
        """Browse for output directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)
    
    def get_config(self) -> TrainingConfig:
        """Get configuration from UI"""
        config = TrainingConfig(
            model_name=self.model_combo.currentText(),
            preset=self.preset_combo.currentText(),
            data_path=self.data_path_edit.text(),
            train_split=self.train_split_spin.value(),
            val_split=self.val_split_spin.value(),
            test_split=self.test_split_spin.value(),
            use_symlinks=self.use_symlinks_check.isChecked(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_size_spin.value(),
            imgsz=int(self.imgsz_combo.currentText()),
            device=self.device_combo.currentText(),
            workers=self.workers_spin.value(),
            resume=self.resume_check.isChecked(),
            optimizer=self.optimizer_combo.currentText(),
            lr0=self.lr0_spin.value(),
            lrf=self.lrf_spin.value(),
            momentum=self.momentum_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            patience=self.patience_spin.value(),
            save_period=self.save_period_spin.value(),
            warmup_epochs=self.warmup_epochs_spin.value(),
            box=self.box_spin.value(),
            cls=self.cls_spin.value(),
            dfl=self.dfl_spin.value(),
            use_augmentation=self.augment_check.isChecked(),
            degrees=self.degrees_spin.value(),
            scale=self.scale_spin.value(),
            fliplr=self.fliplr_spin.value(),
            flipud=self.flipud_spin.value(),
            use_mixed_precision=self.mixed_precision_check.isChecked(),
            multi_scale=self.multi_scale_check.isChecked(),
            rect=self.rect_check.isChecked(),
            cache=self.cache_check.isChecked(),
            close_mosaic=self.close_mosaic_spin.value(),
            overlap_mask=self.overlap_mask_check.isChecked(),
            single_cls=self.single_cls_check.isChecked(),
            dropout=self.dropout_spin.value(),
            cos_lr=self.cos_lr_check.isChecked(),
            project=self.output_dir_edit.text(),
            name=self.run_name_edit.text(),
            exist_ok=self.exist_ok_check.isChecked(),
        )
        return config
    
    def set_config(self, config: TrainingConfig):
        """Set UI from configuration"""
        self.model_combo.setCurrentText(config.model_name)
        self.preset_combo.setCurrentText(config.preset)
        self.data_path_edit.setText(config.data_path)
        self.train_split_spin.setValue(config.train_split)
        self.val_split_spin.setValue(config.val_split)
        self.test_split_spin.setValue(config.test_split)
        self.use_symlinks_check.setChecked(config.use_symlinks)
        self.epochs_spin.setValue(config.epochs)
        self.batch_size_spin.setValue(config.batch_size)
        self.imgsz_combo.setCurrentText(str(config.imgsz))
        self.device_combo.setCurrentText(config.device)
        self.workers_spin.setValue(config.workers)
        self.resume_check.setChecked(config.resume)
        self.optimizer_combo.setCurrentText(config.optimizer)
        self.lr0_spin.setValue(config.lr0)
        self.lrf_spin.setValue(config.lrf)
        self.momentum_spin.setValue(config.momentum)
        self.weight_decay_spin.setValue(config.weight_decay)
        self.patience_spin.setValue(config.patience)
        self.save_period_spin.setValue(config.save_period)
        self.warmup_epochs_spin.setValue(config.warmup_epochs)
        self.box_spin.setValue(config.box)
        self.cls_spin.setValue(config.cls)
        self.dfl_spin.setValue(config.dfl)
        self.augment_check.setChecked(config.use_augmentation)
        self.degrees_spin.setValue(config.degrees)
        self.scale_spin.setValue(config.scale)
        self.fliplr_spin.setValue(config.fliplr)
        self.flipud_spin.setValue(config.flipud)
        self.mixed_precision_check.setChecked(config.use_mixed_precision)
        self.multi_scale_check.setChecked(config.multi_scale)
        self.rect_check.setChecked(config.rect)
        self.cache_check.setChecked(config.cache)
        self.close_mosaic_spin.setValue(config.close_mosaic)
        self.overlap_mask_check.setChecked(config.overlap_mask)
        self.single_cls_check.setChecked(config.single_cls)
        self.dropout_spin.setValue(config.dropout)
        self.cos_lr_check.setChecked(config.cos_lr)
        self.output_dir_edit.setText(config.project)
        self.run_name_edit.setText(config.name)
        self.exist_ok_check.setChecked(config.exist_ok)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.settings = QSettings('YOLOTrainer', 'Settings')
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle("YOLO Training GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        config_label = QLabel("Configuration")
        config_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(config_label)
        
        self.config_widget = ConfigWidget()
        left_layout.addWidget(self.config_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        # Right side - Progress and Logs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Progress section
        progress_label = QLabel("Training Progress")
        progress_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        right_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to start training")
        right_layout.addWidget(self.status_label)
        
        # Logs section
        logs_label = QLabel("Training Logs")
        logs_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(logs_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        right_layout.addWidget(self.log_text)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_toolbar(self):
        """Create toolbar with actions"""
        toolbar = self.addToolBar("Main")
        
        # Load config action
        load_action = toolbar.addAction("Load Config")
        load_action.triggered.connect(self.load_config)
        
        # Save config action
        save_action = toolbar.addAction("Save Config")
        save_action.triggered.connect(self.save_config)
        
        toolbar.addSeparator()
        
        # Export config action
        export_action = toolbar.addAction("Export Config")
        export_action.triggered.connect(self.export_config)
        
        toolbar.addSeparator()
        
        # Clear logs action
        clear_logs_action = toolbar.addAction("Clear Logs")
        clear_logs_action.triggered.connect(self.clear_logs)
        
        # Save logs action
        save_logs_action = toolbar.addAction("Save Logs")
        save_logs_action.triggered.connect(self.save_logs)
        
        toolbar.addSeparator()
        
        # About action
        about_action = toolbar.addAction("About")
        about_action.triggered.connect(self.show_about)
    
    def start_training(self):
        """Start the training process"""
        # Validate configuration
        config = self.config_widget.get_config()
        
        # Check if splits sum to 1.0
        total_split = config.train_split + config.val_split + config.test_split
        if abs(total_split - 1.0) > 0.001:
            QMessageBox.warning(
                self, "Invalid Configuration",
                f"Dataset splits must sum to 1.0 (current: {total_split:.2f})"
            )
            return
        
        # Check dataset path
        if not Path(config.data_path).exists():
            QMessageBox.warning(
                self, "Invalid Dataset Path",
                f"Dataset path does not exist: {config.data_path}"
            )
            return
        
        # Save configuration
        config_path = Path(config.project) / config.name / "training_config.yaml" if config.project else Path("training_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_path))
        self.log(f"Configuration saved to {config_path}")
        
        # Disable controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.config_widget.setEnabled(False)
        
        # Start training thread
        self.training_thread = TrainingThread(config)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.log_update.connect(self.log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.error_signal.connect(self.training_error)
        self.training_thread.start()
        
        self.statusBar().showMessage("Training in progress...")
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Stop",
                "Are you sure you want to stop training?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.training_thread.stop()
                self.training_thread.quit()
                self.training_thread.wait()
                self.log("Training stopped by user")
                self.reset_ui()
    
    def update_progress(self, message: str, progress: int):
        """Update progress bar and status"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def training_finished(self, results: dict):
        """Handle training completion"""
        self.log("\n" + "="*50)
        self.log("TRAINING COMPLETED")
        self.log("="*50)
        self.log(f"Success: {results['success']}")
        self.log(f"Dataset Statistics: {results['dataset_stats']}")
        self.log(f"Training Completed: {results['training_completed']}")
        self.log(f"Models Exported: {results['models_exported']}")
        
        if results['errors']:
            self.log(f"Errors: {', '.join(results['errors'])}")
        
        self.log("="*50)
        
        if results['success']:
            QMessageBox.information(
                self, "Training Complete",
                "Training completed successfully!"
            )
        else:
            QMessageBox.warning(
                self, "Training Complete",
                "Training completed with errors. Check the logs for details."
            )
        
        self.reset_ui()
    
    def training_error(self, error: str):
        """Handle training error"""
        self.log(f"ERROR: {error}")
        QMessageBox.critical(
            self, "Training Error",
            f"An error occurred during training:\n{error}"
        )
        self.reset_ui()
    
    def reset_ui(self):
        """Reset UI after training"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.config_widget.setEnabled(True)
        self.statusBar().showMessage("Ready")
    
    def load_config(self):
        """Load configuration from file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            "", "YAML Files (*.yaml *.yml)"
        )
        
        if path:
            try:
                config = TrainingConfig.from_yaml(path)
                self.config_widget.set_config(config)
                self.log(f"Configuration loaded from {path}")
                QMessageBox.information(
                    self, "Success",
                    "Configuration loaded successfully!"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load configuration:\n{e}"
                )
    
    def save_config(self):
        """Save configuration to file"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            "training_config.yaml", "YAML Files (*.yaml *.yml)"
        )
        
        if path:
            try:
                config = self.config_widget.get_config()
                config.to_yaml(path)
                self.log(f"Configuration saved to {path}")
                QMessageBox.information(
                    self, "Success",
                    "Configuration saved successfully!"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save configuration:\n{e}"
                )
    
    def export_config(self):
        """Export configuration as JSON"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration",
            "training_config.json", "JSON Files (*.json)"
        )
        
        if path:
            try:
                config = self.config_widget.get_config()
                with open(path, 'w') as f:
                    json.dump(asdict(config), f, indent=2)
                self.log(f"Configuration exported to {path}")
                QMessageBox.information(
                    self, "Success",
                    "Configuration exported successfully!"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export configuration:\n{e}"
                )
    
    def clear_logs(self):
        """Clear the log text"""
        self.log_text.clear()
    
    def save_logs(self):
        """Save logs to file"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Logs",
            f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if path:
            try:
                with open(path, 'w') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(
                    self, "Success",
                    "Logs saved successfully!"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save logs:\n{e}"
                )
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About YOLO Training GUI",
            "YOLO Training GUI v1.0\n\n"
            "A user-friendly interface for training YOLO models.\n\n"
            "Features:\n"
            "• Easy configuration management\n"
            "• Real-time training progress monitoring\n"
            "• Support for all YOLO training parameters\n"
            "• Preset configurations for common scenarios\n"
            "• Export models in multiple formats\n\n"
            "Built with PyQt6 and Ultralytics YOLO"
        )
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
    
    def load_settings(self):
        """Load application settings"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "Training in Progress",
                "Training is still in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self.training_thread.stop()
            self.training_thread.quit()
            self.training_thread.wait()
        
        self.save_settings()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()