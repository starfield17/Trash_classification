"""
Main GUI module for YOLO training application.
Provides a PyQt5 interface for configuring and running YOLO training.
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QCheckBox, QTextEdit, QProgressBar,
    QLabel, QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import QThread, Qt, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor

from config import DATAPATH, TRAINING_PROFILES
from training_core import TrainingWorker


class YOLOTrainingGUI(QMainWindow):
    """Main window for YOLO training application"""
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.training_worker = None
        self.datapath = DATAPATH
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("YOLO Training Application")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Data selection section
        data_group = QGroupBox("Dataset Configuration")
        data_layout = QHBoxLayout()
        
        self.data_label = QLabel(f"Data Path: {self.datapath}")
        self.data_label.setWordWrap(True)
        data_layout.addWidget(self.data_label)
        
        self.select_data_btn = QPushButton("Select Data Folder")
        self.select_data_btn.clicked.connect(self.select_data_folder)
        data_layout.addWidget(self.select_data_btn)
        
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
        
        # Training configuration section
        config_group = QGroupBox("Training Configuration")
        config_layout = QVBoxLayout()
        
        # Training profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Training Profile:"))
        
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(TRAINING_PROFILES.keys()))
        self.profile_combo.setCurrentText("default")
        profile_layout.addWidget(self.profile_combo)
        profile_layout.addStretch()
        
        config_layout.addLayout(profile_layout)
        
        # Checkboxes for options
        options_layout = QHBoxLayout()
        
        self.augmentation_checkbox = QCheckBox("Use Data Augmentation")
        self.augmentation_checkbox.setChecked(True)
        options_layout.addWidget(self.augmentation_checkbox)
        
        self.mixed_precision_checkbox = QCheckBox("Use Mixed Precision")
        self.mixed_precision_checkbox.setChecked(True)
        options_layout.addWidget(self.mixed_precision_checkbox)
        
        self.resume_checkbox = QCheckBox("Resume Training")
        self.resume_checkbox.setChecked(False)
        options_layout.addWidget(self.resume_checkbox)
        
        options_layout.addStretch()
        config_layout.addLayout(options_layout)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def select_data_folder(self):
        """Open dialog to select data folder"""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Data Folder",
            self.datapath,
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            self.datapath = folder
            self.data_label.setText(f"Data Path: {self.datapath}")
            self.log_message(f"Selected data folder: {self.datapath}")
            
    def log_message(self, message):
        """Add message to log text area"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    @pyqtSlot()
    def start_training(self):
        """Start the training process in a separate thread"""
        # Validate data path
        if not os.path.exists(self.datapath):
            QMessageBox.warning(
                self,
                "Invalid Path",
                f"The data path does not exist: {self.datapath}"
            )
            return
            
        # Disable start button and enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear previous log
        self.log_text.clear()
        self.log_message("Preparing to start training...")
        
        # Collect training parameters
        config_params = {
            'datapath': self.datapath,
            'config': self.profile_combo.currentText(),
            'use_augmentation': self.augmentation_checkbox.isChecked(),
            'use_mixed_precision': self.mixed_precision_checkbox.isChecked(),
            'resume': self.resume_checkbox.isChecked()
        }
        
        # Create thread and worker
        self.training_thread = QThread()
        self.training_worker = TrainingWorker()
        
        # Move worker to thread
        self.training_worker.moveToThread(self.training_thread)
        
        # Connect signals and slots
        self.training_worker.progress_updated.connect(self.on_progress_update)
        self.training_worker.training_finished.connect(self.on_training_finished)
        self.training_worker.error_occurred.connect(self.on_error_occurred)
        
        # Connect thread signals
        self.training_thread.started.connect(
            lambda: self.training_worker.run_training(config_params)
        )
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_worker.training_finished.connect(self.training_thread.quit)
        self.training_worker.error_occurred.connect(self.training_thread.quit)
        
        # Start the thread
        self.training_thread.start()
        self.statusBar().showMessage("Training in progress...")
        
    @pyqtSlot()
    def stop_training(self):
        """Stop the training process"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Stop",
                "Are you sure you want to stop the training process?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_message("\nStopping training...")
                # Note: Properly stopping YOLO training is complex
                # This is a simplified version
                if self.training_thread:
                    self.training_thread.quit()
                    self.training_thread.wait()
                self.on_training_finished("Training stopped by user")
                
    @pyqtSlot(str)
    def on_progress_update(self, message):
        """Handle progress update from worker"""
        self.log_message(message)
        
    @pyqtSlot(str)
    def on_training_finished(self, message):
        """Handle training completion"""
        self.log_message(f"\n{message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Training completed")
        
        # Clean up worker and thread
        if self.training_worker:
            self.training_worker.deleteLater()
            self.training_worker = None
        if self.training_thread:
            self.training_thread.deleteLater()
            self.training_thread = None
            
        # Show completion dialog
        QMessageBox.information(
            self,
            "Training Complete",
            message
        )
        
    @pyqtSlot(str)
    def on_error_occurred(self, error_message):
        """Handle error from worker"""
        self.log_message(f"\nERROR: {error_message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error occurred")
        
        # Clean up worker and thread
        if self.training_worker:
            self.training_worker.deleteLater()
            self.training_worker = None
        if self.training_thread:
            self.training_thread.deleteLater()
            self.training_thread = None
            
        # Show error dialog
        QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred during training:\n\n{error_message[:500]}..."
        )
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.training_thread:
                    self.training_thread.quit()
                    self.training_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = YOLOTrainingGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
