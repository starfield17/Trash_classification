#!/usr/bin/env python3
"""
YOLO Training CLI Interface
Command-line interface for expert users to train YOLO models with flexible configuration.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml
from train_core import TrainingConfig, TrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLITrainer:
    """CLI handler for YOLO training"""
    
    def __init__(self):
        self.parser = self.create_parser()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all training options"""
        parser = argparse.ArgumentParser(
            description="YOLO Model Training CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Use default configuration
  python train_cli.py
  
  # Use configuration file
  python train_cli.py --config config.yaml
  
  # Use preset configuration
  python train_cli.py --preset servermode
  
  # Override specific parameters
  python train_cli.py --epochs 200 --batch-size 16 --device 0
  
  # Resume training
  python train_cli.py --resume --config last_config.yaml
  
  # Generate template configuration file
  python train_cli.py --generate-config my_config.yaml
            """
        )
        
        # Configuration file options
        config_group = parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration YAML file'
        )
        config_group.add_argument(
            '--generate-config',
            type=str,
            metavar='PATH',
            help='Generate a template configuration file and exit'
        )
        config_group.add_argument(
            '--preset',
            choices=['default', 'large_dataset', 'small_dataset', 
                    'focus_accuracy', 'focus_speed', 'servermode'],
            default='default',
            help='Use preset configuration (default: %(default)s)'
        )
        
        # Model settings
        model_group = parser.add_argument_group('Model Settings')
        model_group.add_argument(
            '--model',
            type=str,
            default='yolo11s.pt',
            help='YOLO model to use (default: %(default)s)'
        )
        
        # Dataset settings
        data_group = parser.add_argument_group('Dataset Settings')
        data_group.add_argument(
            '--data-path',
            type=str,
            default='./label',
            help='Path to dataset directory (default: %(default)s)'
        )
        data_group.add_argument(
            '--train-split',
            type=float,
            default=0.9,
            help='Training data split ratio (default: %(default)s)'
        )
        data_group.add_argument(
            '--val-split',
            type=float,
            default=0.05,
            help='Validation data split ratio (default: %(default)s)'
        )
        data_group.add_argument(
            '--test-split',
            type=float,
            default=0.05,
            help='Test data split ratio (default: %(default)s)'
        )
        data_group.add_argument(
            '--no-symlinks',
            action='store_true',
            help='Copy files instead of using symbolic links'
        )
        
        # Training parameters
        train_group = parser.add_argument_group('Training Parameters')
        train_group.add_argument(
            '--epochs',
            type=int,
            default=120,
            help='Number of training epochs (default: %(default)s)'
        )
        train_group.add_argument(
            '--batch-size', '-b',
            type=int,
            default=10,
            help='Batch size (default: %(default)s)'
        )
        train_group.add_argument(
            '--imgsz', '--img-size',
            type=int,
            default=640,
            help='Input image size (default: %(default)s)'
        )
        train_group.add_argument(
            '--device', '-d',
            type=str,
            default='auto',
            help='Device to use (auto/cpu/0/1/...) (default: %(default)s)'
        )
        train_group.add_argument(
            '--workers', '-w',
            type=int,
            default=-1,
            help='Number of worker threads (-1 for auto) (default: %(default)s)'
        )
        train_group.add_argument(
            '--resume', '-r',
            action='store_true',
            help='Resume training from last checkpoint'
        )
        
        # Optimization parameters
        optim_group = parser.add_argument_group('Optimization Parameters')
        optim_group.add_argument(
            '--optimizer',
            type=str,
            default='AdamW',
            choices=['Adam', 'AdamW', 'SGD', 'RMSProp'],
            help='Optimizer type (default: %(default)s)'
        )
        optim_group.add_argument(
            '--lr0', '--learning-rate',
            type=float,
            default=0.0005,
            help='Initial learning rate (default: %(default)s)'
        )
        optim_group.add_argument(
            '--lrf',
            type=float,
            default=0.01,
            help='Final learning rate ratio (default: %(default)s)'
        )
        optim_group.add_argument(
            '--momentum',
            type=float,
            default=0.937,
            help='Optimizer momentum (default: %(default)s)'
        )
        optim_group.add_argument(
            '--weight-decay',
            type=float,
            default=0.0005,
            help='Weight decay (default: %(default)s)'
        )
        optim_group.add_argument(
            '--patience',
            type=int,
            default=15,
            help='Early stopping patience (default: %(default)s)'
        )
        
        # Loss weights
        loss_group = parser.add_argument_group('Loss Weights')
        loss_group.add_argument(
            '--box',
            type=float,
            default=4.0,
            help='Box loss weight (default: %(default)s)'
        )
        loss_group.add_argument(
            '--cls',
            type=float,
            default=2.0,
            help='Classification loss weight (default: %(default)s)'
        )
        loss_group.add_argument(
            '--dfl',
            type=float,
            default=1.5,
            help='DFL loss weight (default: %(default)s)'
        )
        
        # Augmentation settings
        aug_group = parser.add_argument_group('Data Augmentation')
        aug_group.add_argument(
            '--augment',
            action='store_true',
            help='Enable data augmentation'
        )
        aug_group.add_argument(
            '--degrees',
            type=float,
            default=10.0,
            help='Rotation angle range (default: %(default)s)'
        )
        aug_group.add_argument(
            '--scale',
            type=float,
            default=0.2,
            help='Scaling ratio range (default: %(default)s)'
        )
        aug_group.add_argument(
            '--fliplr',
            type=float,
            default=0.2,
            help='Horizontal flip probability (default: %(default)s)'
        )
        aug_group.add_argument(
            '--flipud',
            type=float,
            default=0.2,
            help='Vertical flip probability (default: %(default)s)'
        )
        
        # Advanced settings
        advanced_group = parser.add_argument_group('Advanced Settings')
        advanced_group.add_argument(
            '--mixed-precision', '--fp16',
            action='store_true',
            help='Enable mixed precision training'
        )
        advanced_group.add_argument(
            '--multi-scale',
            action='store_true',
            default=True,
            help='Enable multi-scale training (default: %(default)s)'
        )
        advanced_group.add_argument(
            '--rect',
            action='store_true',
            default=True,
            help='Enable rectangular training (default: %(default)s)'
        )
        advanced_group.add_argument(
            '--cache',
            action='store_true',
            default=True,
            help='Cache images in memory (default: %(default)s)'
        )
        advanced_group.add_argument(
            '--dropout',
            type=float,
            default=0.0,
            help='Dropout rate (default: %(default)s)'
        )
        advanced_group.add_argument(
            '--cos-lr',
            action='store_true',
            help='Use cosine learning rate scheduler'
        )
        
        # Output settings
        output_group = parser.add_argument_group('Output Settings')
        output_group.add_argument(
            '--project',
            type=str,
            default='',
            help='Project directory for results'
        )
        output_group.add_argument(
            '--name',
            type=str,
            default='runs/train',
            help='Run name (default: %(default)s)'
        )
        output_group.add_argument(
            '--exist-ok',
            action='store_true',
            default=True,
            help='Allow overwriting existing results (default: %(default)s)'
        )
        
        # Utility options
        utility_group = parser.add_argument_group('Utility Options')
        utility_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        utility_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress output except errors'
        )
        utility_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show configuration without training'
        )
        
        return parser
    
    def load_config(self, args) -> TrainingConfig:
        """Load configuration from file or command line arguments"""
        # Start with default config
        config = TrainingConfig()
        
        # Load from config file if provided
        if args.config:
            try:
                config = TrainingConfig.from_yaml(args.config)
                logger.info(f"Loaded configuration from {args.config}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
                sys.exit(1)
        
        # Override with command line arguments
        cli_overrides = {
            'model_name': args.model,
            'data_path': args.data_path,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'test_split': args.test_split,
            'use_symlinks': not args.no_symlinks,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'imgsz': args.imgsz,
            'device': args.device,
            'workers': args.workers,
            'resume': args.resume,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'box': args.box,
            'cls': args.cls,
            'dfl': args.dfl,
            'use_augmentation': args.augment,
            'degrees': args.degrees,
            'scale': args.scale,
            'fliplr': args.fliplr,
            'flipud': args.flipud,
            'use_mixed_precision': args.mixed_precision,
            'multi_scale': args.multi_scale,
            'rect': args.rect,
            'cache': args.cache,
            'dropout': args.dropout,
            'cos_lr': args.cos_lr,
            'project': args.project,
            'name': args.name,
            'exist_ok': args.exist_ok,
            'preset': args.preset,
        }
        
        # Apply overrides (only if different from defaults)
        for key, value in cli_overrides.items():
            if value != getattr(TrainingConfig, key, None):
                setattr(config, key, value)
        
        # Apply preset if specified
        if config.preset != 'default':
            config.apply_preset()
            logger.info(f"Applied preset configuration: {config.preset}")
        
        return config
    
    def generate_config_template(self, path: str):
        """Generate a template configuration file"""
        config = TrainingConfig()
        config.to_yaml(path)
        logger.info(f"Generated configuration template at {path}")
        print(f"\nConfiguration template saved to: {path}")
        print("Edit this file to customize your training parameters.")
    
    def print_config(self, config: TrainingConfig):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        
        sections = {
            "Model": ['model_name', 'preset'],
            "Dataset": ['data_path', 'train_split', 'val_split', 'test_split'],
            "Training": ['epochs', 'batch_size', 'imgsz', 'device', 'workers'],
            "Optimization": ['optimizer', 'lr0', 'lrf', 'weight_decay', 'patience'],
            "Loss Weights": ['box', 'cls', 'dfl'],
            "Augmentation": ['use_augmentation', 'degrees', 'scale', 'fliplr', 'flipud'],
            "Advanced": ['use_mixed_precision', 'multi_scale', 'dropout', 'cos_lr'],
            "Output": ['project', 'name', 'resume']
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = getattr(config, key, None)
                if value is not None:
                    print(f"  {key:20s}: {value}")
        
        print("\n" + "="*60 + "\n")
    
    def progress_callback(self, message: str, progress: int):
        """Progress callback for training pipeline"""
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r{message:50s} [{bar}] {progress:3d}%', end='', flush=True)
        if progress == 100:
            print()  # New line when complete
    
    def run(self):
        """Main CLI execution"""
        args = self.parser.parse_args()
        
        # Set logging level
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Generate config template if requested
        if args.generate_config:
            self.generate_config_template(args.generate_config)
            return 0
        
        # Load configuration
        config = self.load_config(args)
        
        # Validate configuration
        if config.train_split + config.val_split + config.test_split != 1.0:
            logger.error("Dataset splits must sum to 1.0")
            return 1
        
        # Print configuration
        if not args.quiet:
            self.print_config(config)
        
        # Dry run - just show config
        if args.dry_run:
            print("DRY RUN - No training will be performed")
            return 0
        
        # Save configuration for reproducibility
        config_save_path = Path(config.project) / config.name / "training_config.yaml" if config.project else Path("training_config.yaml")
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_save_path))
        logger.info(f"Configuration saved to {config_save_path}")
        
        # Run training pipeline
        try:
            print("\nStarting training pipeline...")
            pipeline = TrainingPipeline(config)
            
            # Use progress callback only if not quiet
            callback = self.progress_callback if not args.quiet else None
            results = pipeline.run(progress_callback=callback)
            
            # Print results
            if not args.quiet:
                print("\n" + "="*60)
                print("TRAINING RESULTS")
                print("="*60)
                print(f"Success: {results['success']}")
                print(f"Dataset Statistics: {results['dataset_stats']}")
                print(f"Training Completed: {results['training_completed']}")
                print(f"Models Exported: {results['models_exported']}")
                
                if results['errors']:
                    print(f"Errors: {', '.join(results['errors'])}")
                
                print("="*60 + "\n")
            
            return 0 if results['success'] else 1
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1


def main():
    """Main entry point"""
    cli = CLITrainer()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()