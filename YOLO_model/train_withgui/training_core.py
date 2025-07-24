"""
Core training module for YOLO training application.
Contains the TrainingWorker class that handles training in a separate thread.
"""

import os
import gc
import shutil
import torch
import traceback
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal
from ultralytics import YOLO

from config import (
    SELECT_MODEL, DEFAULT_TRAIN_CONFIG, TRAINING_PROFILES,
    AUGMENTATION_CONFIG
)
from data_utils import (
    check_and_clean_dataset, create_data_yaml, prepare_dataset
)


class TrainingWorker(QObject):
    """Worker class for YOLO training that runs in a separate thread"""
    
    # Define signals
    progress_updated = pyqtSignal(str)  # For status updates and log messages
    training_finished = pyqtSignal(str)  # For final result message
    error_occurred = pyqtSignal(str)    # For error messages
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.is_running = False
        
    def emit_progress(self, message):
        """Helper method to emit progress updates"""
        self.progress_updated.emit(message)
        
    def run_training(self, config_params):
        """
        Main execution method that runs the complete training pipeline.
        
        Args:
            config_params (dict): Dictionary containing all training parameters
                - datapath: Path to dataset
                - config: Training profile name
                - use_augmentation: Boolean for data augmentation
                - use_mixed_precision: Boolean for mixed precision training
                - resume: Boolean for resuming training
        """
        self.is_running = True
        
        try:
            # Clear GPU cache at start
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract parameters
            data_dir = config_params.get('datapath', './label')
            training_config = config_params.get('config', 'default')
            use_augmentation = config_params.get('use_augmentation', False)
            use_mixed_precision = config_params.get('use_mixed_precision', False)
            resume = config_params.get('resume', False)
            
            # Step 1: Check dataset
            self.emit_progress("Step 1: Checking dataset...")
            valid_pairs = check_and_clean_dataset(data_dir)
            
            if not valid_pairs:
                self.error_occurred.emit("No valid data pairs found. Training cannot proceed.")
                return
            
            gc.collect()
            
            # Step 2: Create configuration file
            self.emit_progress("\nStep 2: Creating data.yaml...")
            create_data_yaml()
            project_path = os.path.dirname(os.path.abspath(__file__))
            data_yaml_path = os.path.join(project_path, "data.yaml")
            
            # Step 3: Prepare dataset
            self.emit_progress("\nStep 3: Preparing dataset with symbolic links...")
            try:
                train_size, val_size, test_size = prepare_dataset(
                    data_dir, valid_pairs, use_symlinks=True
                )
                gc.collect()
                
                if val_size < 5:
                    self.emit_progress(
                        f"Warning: Validation set size ({val_size}) is less than 5. "
                        "INT8 calibration might be suboptimal."
                    )
            except ValueError as ve:
                self.error_occurred.emit(f"Error during dataset preparation: {ve}")
                return
            
            # Step 4: Start training
            self.emit_progress("\nStep 4: Starting training...")
            results = self.train_yolo(
                use_augmentation=use_augmentation,
                use_mixed_precision=use_mixed_precision,
                config=training_config,
                resume=resume
            )
            
            if results:
                # Step 5: Save different precision models
                self.emit_progress("\nStep 5: Saving different precision models based on best.pt...")
                
                weights_dir = None
                if hasattr(results, 'save_dir'):
                    weights_dir = os.path.join(results.save_dir, 'weights')
                    self.emit_progress(f"Found weights directory from results: {weights_dir}")
                else:
                    # Fallback: Find the latest run directory
                    default_project = project_path
                    default_run_name_base = "runs/train"
                    
                    run_dirs = sorted(
                        Path(default_project).glob(f"{Path(default_run_name_base).name}*/"),
                        key=os.path.getmtime,
                        reverse=True
                    )
                    if run_dirs:
                        latest_run_dir = run_dirs[0]
                        weights_dir = os.path.join(latest_run_dir, 'weights')
                        self.emit_progress(f"Constructed weights directory (fallback): {weights_dir}")
                    else:
                        self.emit_progress("Warning: Could not determine weights directory automatically.")
                
                if weights_dir and os.path.isdir(weights_dir):
                    self.save_quantized_models(weights_dir, data_yaml_path)
                else:
                    self.emit_progress(
                        f"Warning: Weights directory '{weights_dir}' not found. "
                        "Skipping post-training save."
                    )
                
                self.training_finished.emit("Training completed successfully!")
            else:
                self.error_occurred.emit(
                    "Training did not complete successfully or was interrupted."
                )
                
        except Exception as e:
            error_msg = f"An error occurred in the training pipeline: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            # Final cleanup
            self.emit_progress("\nScript finished. Cleaning up...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_running = False
    
    def train_yolo(self, use_augmentation=False, use_mixed_precision=False, 
                   config="default", resume=False):
        """
        YOLO training method with configurable options.
        """
        try:
            self.model = YOLO(SELECT_MODEL)
            num_workers = max(1, min(os.cpu_count() - 2, 8))
            device = "cpu"
            
            if torch.cuda.is_available():
                device = "0"
                
            if device == "cpu":
                batch_size = 4
                workers = max(1, min(os.cpu_count() - 1, 4))
                use_mixed_precision = False
            else:
                batch_size = 10
                workers = num_workers
            
            # Start with default configuration
            train_args = DEFAULT_TRAIN_CONFIG.copy()
            train_args.update({
                "data": "data.yaml",
                "batch": batch_size,
                "workers": workers,
                "device": device,
                "project": os.path.dirname(os.path.abspath(__file__)),
                "name": "runs/train",
                "resume": resume
            })
            
            # Apply profile-specific configurations
            if config in TRAINING_PROFILES and config != "default":
                profile_config = TRAINING_PROFILES[config].copy()
                
                # Adjust batch size based on device
                if "batch" in profile_config:
                    if device == "cpu":
                        profile_config["batch"] = 4
                    
                train_args.update(profile_config)
                
            elif config != "default":
                self.emit_progress(
                    f"Warning: Unrecognized configuration mode '{config}', "
                    "using default configuration."
                )
            
            # Apply augmentation settings
            if use_augmentation:
                train_args.update(AUGMENTATION_CONFIG["enabled"])
            else:
                train_args.update(AUGMENTATION_CONFIG["disabled"])
            
            # Enable mixed-precision training (only on GPU)
            if use_mixed_precision and device == "0":
                train_args.update({"half": True})
            else:
                train_args.update({"half": False})
            
            self.emit_progress(f"\nUsing device: {'GPU' if device == '0' else 'CPU'}")
            self.emit_progress(f"Batch size: {train_args['batch']}")
            self.emit_progress(
                f"Mixed-precision training: "
                f"{'Enabled' if train_args.get('half', False) else 'Disabled'}\n"
            )
            
            results = self.model.train(**train_args)
            
            # Handle resume mode
            if resume:
                self.emit_progress(
                    "\nDetected resume=True, ensuring final model is saved to best.pt..."
                )
                run_dir = results.save_dir if hasattr(results, 'save_dir') else (
                    train_args.get('project', '') + '/' + train_args.get('name', 'runs/train')
                )
                weights_dir = os.path.join(run_dir, 'weights')
                last_pt_path = os.path.join(weights_dir, 'last.pt')
                best_pt_path = os.path.join(weights_dir, 'best.pt')
                
                if os.path.exists(last_pt_path):
                    if not os.path.exists(best_pt_path) or resume:
                        self.emit_progress(f"Copying {last_pt_path} to {best_pt_path}...")
                        shutil.copy2(last_pt_path, best_pt_path)
                        self.emit_progress(f"Successfully saved final model to {best_pt_path}")
            
            return results
            
        except Exception as e:
            self.emit_progress(f"Training error: {str(e)}")
            self.emit_progress(traceback.format_exc())
            return None
    
    def save_quantized_models(self, weights_dir, data_yaml_path):
        """Load the best model and save FP16 version"""
        best_pt_path = os.path.join(weights_dir, 'best.pt')
        
        if not os.path.exists(best_pt_path):
            self.emit_progress(
                f"Error: {best_pt_path} not found, cannot save quantized model."
            )
            return
        
        self.emit_progress(f"\nLoading best model from {best_pt_path}...")
        
        try:
            model = YOLO(best_pt_path)
        except Exception as e:
            self.emit_progress(f"Error loading model {best_pt_path}: {e}")
            return
        
        # Create FP16 model
        self.emit_progress("\nCreating FP16 model...")
        fp16_model_path = os.path.join(weights_dir, 'best_fp16.pt')
        shutil.copy(best_pt_path, fp16_model_path)
        
        # Load copy and convert to FP16
        fp16_model = YOLO(fp16_model_path)
        if hasattr(fp16_model, 'model'):
            fp16_model.model = fp16_model.model.half()
            fp16_model.save(fp16_model_path)
            self.emit_progress(f"FP16 model saved to {fp16_model_path}")
        else:
            self.emit_progress(
                "Cannot convert to FP16 model: model structure not as expected"
            )
        
        # Clean up memory
        del fp16_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Export TorchScript model (optional)
        try:
            self.emit_progress("\nAttempting to export TorchScript format model...")
            torchscript_results = model.export(format='torchscript')
            
            if hasattr(torchscript_results, 'saved_model'):
                ts_path = torchscript_results.saved_model
                self.emit_progress(f"TorchScript model exported to: {ts_path}")
                
                # Move to target location if needed
                if os.path.dirname(ts_path) != weights_dir:
                    ts_target_path = os.path.join(weights_dir, 'best.torchscript')
                    shutil.copy(ts_path, ts_target_path)
                    self.emit_progress(f"TorchScript model copied to: {ts_target_path}")
        except Exception as e:
            self.emit_progress(f"TorchScript export failed: {e}")
        
        self.emit_progress("\nModel export and save operations complete!")
        
        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
