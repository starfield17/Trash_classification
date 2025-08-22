"""
YOLO Training Core Module
This module contains the core functionality for YOLO model training,
dataset preparation, and model export.
"""

import os
import json
import yaml
import shutil
import cv2
import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for YOLO training parameters"""
    # Model settings
    model_name: str = "yolo11s.pt"
    
    # Dataset settings
    data_path: str = "./label"
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    use_symlinks: bool = True
    
    # Training parameters
    epochs: int = 120
    batch_size: int = 10
    imgsz: int = 640
    device: str = "auto"  # auto, cpu, or cuda device number
    workers: int = -1  # -1 for auto
    
    # Optimization parameters
    optimizer: str = "AdamW"
    lr0: float = 0.0005
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Loss weights
    box: float = 4.0
    cls: float = 2.0
    dfl: float = 1.5
    
    # Training options
    patience: int = 15
    save_period: int = 5
    warmup_epochs: int = 10
    warmup_momentum: float = 0.5
    warmup_bias_lr: float = 0.05
    
    # Augmentation settings
    use_augmentation: bool = False
    degrees: float = 10.0
    scale: float = 0.2
    fliplr: float = 0.2
    flipud: float = 0.2
    
    # Advanced settings
    use_mixed_precision: bool = False
    multi_scale: bool = True
    rect: bool = True
    cache: bool = True
    close_mosaic: int = 0
    overlap_mask: bool = False
    single_cls: bool = False
    dropout: float = 0.0
    cos_lr: bool = False
    
    # Output settings
    project: str = ""
    name: str = "runs/train"
    exist_ok: bool = True
    resume: bool = False
    
    # Preset configurations
    preset: str = "default"  # default, large_dataset, small_dataset, focus_accuracy, focus_speed, servermode
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def apply_preset(self):
        """Apply preset configurations"""
        if self.preset == "large_dataset":
            self.batch_size = 32 if "cuda" in str(self.device) else 4
            self.lr0 = 0.001
            self.epochs = 150
            self.patience = 30
        elif self.preset == "small_dataset":
            self.batch_size = 16 if "cuda" in str(self.device) else 4
            self.lr0 = 0.0001
            self.weight_decay = 0.001
            self.warmup_epochs = 15
        elif self.preset == "focus_accuracy":
            self.imgsz = 640
            self.box = 7.5
            self.cls = 4.0
            self.dfl = 3.0
            self.patience = 20
            self.batch_size = 16 if "cuda" in str(self.device) else 4
            self.epochs = 300
            self.lr0 = 0.001
            self.lrf = 0.01
            self.weight_decay = 0.0005
            self.dropout = 0.1
        elif self.preset == "focus_speed":
            self.imgsz = 512
            self.epochs = 150
            self.patience = 30
            self.batch_size = 48 if "cuda" in str(self.device) else 4
        elif self.preset == "servermode":
            self.box = 7.5
            self.cls = 4.0
            self.dfl = 3.0
            self.patience = 15
            self.epochs = 150
            self.dropout = 0.1
            self.imgsz = 640
            self.batch_size = 32
            self.lr0 = 0.001
            self.lrf = 0.01
            self.weight_decay = 0.0005
            self.workers = max(1, os.cpu_count() // 2)
            self.device = "0" if torch.cuda.is_available() else "cpu"
            self.use_mixed_precision = True
            self.cache = True
            self.cos_lr = True
            self.overlap_mask = True
            self.multi_scale = True


class DatasetProcessor:
    """Handles dataset validation, preparation, and conversion"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.category_mapping = {
            # Kitchen Waste (0)
            "Kitchen_waste": 0, "potato": 0, "daikon": 0, "carrot": 0,
            # Recyclable Waste (1)
            "Recyclable_waste": 1, "bottle": 1, "can": 1,
            # Hazardous Waste (2)
            "Hazardous_waste": 2, "battery": 2, "drug": 2, "inner_packing": 2,
            # Other Waste (3)
            "Other_waste": 3, "tile": 3, "stone": 3, "brick": 3,
        }
    
    def validate_json_file(self, json_path: str) -> bool:
        """Validates if a JSON file is valid"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
            logger.warning(f"Invalid JSON file {json_path}: {e}")
            return False
    
    def check_single_file(self, img_file: str, data_dir: str) -> Optional[str]:
        """Checks a single image file and its corresponding JSON label file"""
        img_path = os.path.join(data_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        json_file = os.path.join(data_dir, base_name + ".json")
        
        # Check image integrity
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Corrupted or invalid image file: {img_file}")
                return None
            height, width = img.shape[:2]
            if height < 10 or width < 10:
                logger.warning(f"Image dimensions too small: {img_file}")
                return None
        except Exception as e:
            logger.warning(f"Error reading image {img_file}: {e}")
            return None
        
        # Check label file
        if not os.path.exists(json_file):
            logger.warning(f"Corresponding label file not found: {json_file}")
            return None
        
        if not self.validate_json_file(json_file):
            return None
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            if "labels" not in label_data:
                logger.warning(f"Invalid label file structure: {json_file}")
                return None
            return img_file
        except Exception as e:
            logger.warning(f"Error processing label file {json_file}: {e}")
            return None
    
    def check_and_clean_dataset(self) -> List[str]:
        """Checks dataset integrity and returns list of valid image files"""
        logger.info("Checking dataset integrity...")
        data_dir = self.config.data_path
        
        if not os.path.isdir(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return []
        
        image_extensions = (".jpg", ".jpeg", ".png")
        all_files = os.listdir(data_dir)
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        
        logger.info(f"Found {len(image_files)} potential image files")
        if not image_files:
            logger.warning("No supported image files found")
            return []
        
        valid_pairs = []
        max_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.check_single_file, img_file, data_dir) 
                      for img_file in image_files]
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        valid_pairs.append(result)
                except Exception as exc:
                    logger.error(f'File check task exception: {exc}')
        
        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        return valid_pairs
    
    def convert_bbox_to_yolo(self, bbox: Dict, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box from x1,y1,x2,y2 to YOLO format"""
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Ensure values are within 0-1 range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return x_center, y_center, width, height
    
    def convert_labels(self, json_file: str, txt_file: str) -> bool:
        """Convert JSON labels to YOLO format"""
        try:
            base_name = os.path.splitext(json_file)[0]
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                temp_path = base_name + ext
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if not img_path:
                logger.warning(f"No corresponding image for: {json_file}")
                return False
            
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                return False
            
            img_height, img_width = img.shape[:2]
            
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            with open(txt_file, "w", encoding="utf-8") as f:
                if "labels" not in data:
                    logger.warning(f"No 'labels' key in {json_file}")
                    return False
                
                for label in data["labels"]:
                    try:
                        class_name = label.get("name")
                        if class_name not in self.category_mapping:
                            logger.warning(f"Unknown class {class_name} in {json_file}")
                            continue
                        
                        category_id = self.category_mapping[class_name]
                        x_center, y_center, width, height = self.convert_bbox_to_yolo(
                            label, img_width, img_height
                        )
                        
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    except Exception as e:
                        logger.warning(f"Error processing label in {json_file}: {e}")
                        continue
            return True
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            return False
    
    def prepare_dataset(self, valid_pairs: List[str]) -> Tuple[int, int, int]:
        """Prepare dataset splits"""
        if len(valid_pairs) < 15:
            raise ValueError(f"Insufficient data pairs ({len(valid_pairs)}). Need at least 15.")
        
        # Clean existing directories
        for split in ["train", "val", "test"]:
            split_path = Path(self.config.project) / split if self.config.project else Path(split)
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)
            (split_path / "images").mkdir(exist_ok=True)
            (split_path / "labels").mkdir(exist_ok=True)
        
        # Split dataset
        test_size = self.config.val_split + self.config.test_split
        train_files, temp = train_test_split(valid_pairs, test_size=test_size, random_state=42)
        val_ratio = self.config.val_split / test_size
        val_files, test_files = train_test_split(temp, test_size=1-val_ratio, random_state=42)
        
        splits = {"train": train_files, "val": val_files, "test": test_files}
        
        for split_name, files in splits.items():
            self.process_split(split_name, files)
        
        logger.info(f"Dataset prepared - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        return len(train_files), len(val_files), len(test_files)
    
    def process_split(self, split_name: str, files: List[str]):
        """Process a dataset split"""
        split_path = Path(self.config.project) / split_name if self.config.project else Path(split_name)
        split_img_dir = split_path / "images"
        split_lbl_dir = split_path / "labels"
        
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            src_img = Path(self.config.data_path) / img_file
            src_json = Path(self.config.data_path) / f"{base_name}.json"
            dst_img = split_img_dir / img_file
            dst_txt = split_lbl_dir / f"{base_name}.txt"
            
            if src_img.exists() and src_json.exists():
                try:
                    # Copy or link image
                    if self.config.use_symlinks:
                        try:
                            os.symlink(src_img.absolute(), dst_img)
                        except:
                            shutil.copy2(src_img, dst_img)
                    else:
                        shutil.copy2(src_img, dst_img)
                    
                    # Convert labels
                    self.convert_labels(str(src_json), str(dst_txt))
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")


class YOLOTrainer:
    """YOLO model training and export handler"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.results = None
        
    def setup_device(self):
        """Setup training device"""
        if self.config.device == "auto":
            self.config.device = "0" if torch.cuda.is_available() else "cpu"
        if self.config.device == "0":
            self.config.device = "0"
        if self.config.device == "1":
            self.config.device = "1"
        if self.config.device == "2":
            self.config.device = "2"
        if self.config.device == "3":
            self.config.device = "3"
        if self.config.device == "cpu":
            self.config.batch_size = min(self.config.batch_size, 4)
            self.config.use_mixed_precision = False
        if self.config.workers == -1:
            self.config.workers = max(1, min(os.cpu_count() - 2, 8))
    
    def create_data_yaml(self) -> str:
        """Create YOLO data configuration file"""
        current_dir = Path(self.config.project) if self.config.project else Path.cwd()
        data = {
            "path": str(current_dir),
            "train": str(current_dir / "train/images"),
            "val": str(current_dir / "val/images"),
            "test": str(current_dir / "test/images"),
            "names": {
                0: "Kitchen Waste",
                1: "Recyclable Waste",
                2: "Hazardous Waste",
                3: "Other Waste"
            },
            "nc": 4
        }
        
        yaml_path = current_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        
        return str(yaml_path)
    
    def prepare_training_args(self) -> Dict[str, Any]:
        """Prepare training arguments"""
        self.setup_device()
        
        if self.config.preset != "default":
            self.config.apply_preset()
        
        train_args = {
            "data": self.create_data_yaml(),
            "epochs": self.config.epochs,
            "imgsz": self.config.imgsz,
            "batch": self.config.batch_size,
            "workers": self.config.workers,
            "device": self.config.device,
            "patience": self.config.patience,
            "save_period": self.config.save_period,
            "exist_ok": self.config.exist_ok,
            "project": self.config.project or str(Path.cwd()),
            "name": self.config.name,
            "optimizer": self.config.optimizer,
            "lr0": self.config.lr0,
            "lrf": self.config.lrf,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_momentum": self.config.warmup_momentum,
            "warmup_bias_lr": self.config.warmup_bias_lr,
            "box": self.config.box,
            "cls": self.config.cls,
            "dfl": self.config.dfl,
            "close_mosaic": self.config.close_mosaic,
            "nbs": 64,
            "overlap_mask": self.config.overlap_mask,
            "multi_scale": self.config.multi_scale,
            "single_cls": self.config.single_cls,
            "rect": self.config.rect,
            "cache": self.config.cache,
            "resume": self.config.resume,
            "half": self.config.use_mixed_precision and self.config.device != "cpu",
        }
        
        # Add augmentation parameters
        if self.config.use_augmentation:
            train_args.update({
                "augment": True,
                "degrees": self.config.degrees,
                "scale": self.config.scale,
                "fliplr": self.config.fliplr,
                "flipud": self.config.flipud,
            })
        else:
            train_args.update({
                "augment": False,
                "degrees": 0.0,
                "scale": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
            })
        
        # Add optional parameters
        if self.config.dropout > 0:
            train_args["dropout"] = self.config.dropout
        if self.config.cos_lr:
            train_args["cos_lr"] = self.config.cos_lr
        
        return train_args
    
    def train(self, progress_callback=None) -> bool:
        """Train the YOLO model"""
        try:
            self.model = YOLO(self.config.model_name)
            train_args = self.prepare_training_args()
            
            logger.info(f"Starting training with device: {self.config.device}")
            logger.info(f"Batch size: {train_args['batch']}")
            logger.info(f"Mixed-precision: {'Enabled' if train_args.get('half', False) else 'Disabled'}")
            
            self.results = self.model.train(**train_args)
            
            # Handle resume mode
            if self.config.resume and self.results:
                self._handle_resume_save()
            
            return True
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def _handle_resume_save(self):
        """Handle saving when training is resumed"""
        if not self.results:
            return
        
        weights_dir = Path(self.results.save_dir) / 'weights' if hasattr(self.results, 'save_dir') else None
        if not weights_dir:
            weights_dir = Path(self.config.project) / self.config.name / 'weights'
        
        last_pt = weights_dir / 'last.pt'
        best_pt = weights_dir / 'best.pt'
        
        if last_pt.exists() and (not best_pt.exists() or self.config.resume):
            shutil.copy2(last_pt, best_pt)
            logger.info(f"Saved final model to {best_pt}")
    
    def export_models(self, weights_dir: Optional[Path] = None):
        """Export models in different formats"""
        if not weights_dir:
            if self.results and hasattr(self.results, 'save_dir'):
                weights_dir = Path(self.results.save_dir) / 'weights'
            else:
                weights_dir = Path(self.config.project) / self.config.name / 'weights'
        
        best_pt = weights_dir / 'best.pt'
        if not best_pt.exists():
            logger.error(f"Best model not found at {best_pt}")
            return
        
        logger.info(f"Loading best model from {best_pt}")
        model = YOLO(str(best_pt))
        
        # Export FP16 model
        try:
            fp16_path = weights_dir / 'best_fp16.pt'
            shutil.copy2(best_pt, fp16_path)
            fp16_model = YOLO(str(fp16_path))
            if hasattr(fp16_model, 'model'):
                fp16_model.model = fp16_model.model.half()
                fp16_model.save(str(fp16_path))
                logger.info(f"FP16 model saved to {fp16_path}")
        except Exception as e:
            logger.error(f"Failed to create FP16 model: {e}")
        
        # Export TorchScript
        try:
            logger.info("Exporting TorchScript model...")
            ts_results = model.export(format='torchscript')
            logger.info("TorchScript export complete")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrainingPipeline:
    """Complete training pipeline orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = DatasetProcessor(config)
        self.trainer = YOLOTrainer(config)
    
    def run(self, progress_callback=None) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        results = {
            "success": False,
            "dataset_stats": {},
            "training_completed": False,
            "models_exported": False,
            "errors": []
        }
        
        try:
            # Step 1: Check dataset
            if progress_callback:
                progress_callback("Checking dataset integrity...", 10)
            valid_pairs = self.processor.check_and_clean_dataset()
            if not valid_pairs:
                results["errors"].append("No valid data pairs found")
                return results
            
            # Step 2: Prepare dataset
            if progress_callback:
                progress_callback("Preparing dataset splits...", 30)
            train_size, val_size, test_size = self.processor.prepare_dataset(valid_pairs)
            results["dataset_stats"] = {
                "train": train_size,
                "validation": val_size,
                "test": test_size,
                "total": len(valid_pairs)
            }
            
            # Step 3: Train model
            if progress_callback:
                progress_callback("Training model...", 50)
            success = self.trainer.train(progress_callback)
            results["training_completed"] = success
            
            # Step 4: Export models
            if success:
                if progress_callback:
                    progress_callback("Exporting models...", 90)
                self.trainer.export_models()
                results["models_exported"] = True
            
            results["success"] = True
            if progress_callback:
                progress_callback("Training complete!", 100)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(str(e))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results