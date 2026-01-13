"""
YOLO-SCEMA Training Script

Main training script for the YOLO-SCEMA model.
Implements the complete training pipeline including data loading,
model training, validation, and checkpointing.

Author: Mohammed MAIZA
Repository: https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention
Paper: "Efficient Multiscale Attention with Spatial–Channel Reconstruction for Lightweight Object Detection"
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import create_model, YOLO_SCEMA
from data import create_dataloader, ExDarkDataset, VisDroneDataset, FYPDataset, COCODataset, VOCDataset
from data.transforms import TrainTransforms, ValTransforms
from utils.losses import YOLOSCEMALoss
from utils.metrics import DetectionMetrics, ModelEvaluator
from utils.logger import Logger, ProgressLogger
from utils.visualizer import DetectionVisualizer

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    """YOLO-SCEMA Trainer"""
    
    def __init__(self, config_path):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Set random seed
        set_seed(self.config['train']['seed'])
        
        # Setup device
        self.device = self.setup_device()
        print(f"Using device: {self.device}")
        
        # Create experiment directory
        self.exp_dir = self.create_experiment_dir()
        
        # Initialize logger
        self.logger = self.setup_logger()
        
        # Build model
        self.model = self.build_model()
        
        # Build datasets and dataloaders
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer()
        
        # Setup loss function
        self.criterion = self.setup_loss()
        
        # Setup EMA (Exponential Moving Average)
        self.ema = self.setup_ema() if self.config['train']['tricks']['ema_enabled'] else None
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.config['train']['tricks']['mixed_precision'] else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mAP': []
        }
        
        # Load checkpoint if specified
        if 'checkpoint' in self.config['train'] and self.config['train']['checkpoint']:
            self.load_checkpoint(self.config['train']['checkpoint'])
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values if not present
        config.setdefault('train', {})
        config['train'].setdefault('epochs', 300)
        config['train'].setdefault('batch_size', 16)
        config['train'].setdefault('device', 'cuda')
        config['train'].setdefault('seed', 42)
        
        return config
    
    def setup_device(self):
        """Setup training device"""
        device = self.config['train']['device']
        
        if device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def create_experiment_dir(self):
        """Create experiment directory for saving checkpoints and logs"""
        exp_name = self.config['train'].get('experiment_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        exp_dir = Path(self.config['train']['checkpoint']['save_dir']) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_save_path = exp_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return exp_dir
    
    def setup_logger(self):
        """Setup logger for experiment tracking"""
        logger_config = self.config.get('logging', {})
        return Logger(
            log_dir=str(self.exp_dir),
            experiment_name=self.exp_dir.name,
            use_tensorboard=logger_config.get('loggers', {}).get('tensorboard', True),
            use_csv=logger_config.get('loggers', {}).get('csv', True),
            use_json=logger_config.get('loggers', {}).get('json', True)
        )
    
    def build_model(self):
        """Build YOLO-SCEMA model"""
        model_config = self.config['model']
        train_config = self.config['train']
        
        # Get model size
        model_size = model_config.get('size', 'n')
        
        # Get number of classes
        if 'head' in model_config and 'num_classes' in model_config['head']:
            num_classes = model_config['head']['num_classes']
        else:
            # Get from dataset config
            dataset_name = self.config['data'].get('dataset', 'coco')
            num_classes = self.config['datasets'][dataset_name]['num_classes']
        
        # Build model
        model = create_model(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Log model info
        print(f"Model: YOLO-SCEMA-{model_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Log model graph to TensorBoard
        if self.logger.use_tensorboard:
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            self.logger.log_model_graph(model, dummy_input)
        
        return model
    
    def build_dataloaders(self):
        """Build training and validation dataloaders"""
        data_config = self.config['data']
        train_config = self.config['train']
        aug_config = data_config.get('augmentation', {})
        
        # Get dataset name
        dataset_name = data_config.get('dataset', 'coco')
        dataset_path = Path(data_config['dataset_path'])
        
        # Get image size
        img_size = data_config.get('img_size', [640, 640])
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        # Create transforms
        train_transforms = TrainTransforms(
            img_size=img_size,
            hsv_h=aug_config.get('hsv_h', 0.015),
            hsv_s=aug_config.get('hsv_s', 0.7),
            hsv_v=aug_config.get('hsv_v', 0.4),
            degrees=aug_config.get('degrees', 0.0),
            translate=aug_config.get('translate', 0.1),
            scale=aug_config.get('scale', 0.9),
            shear=aug_config.get('shear', 0.0),
            perspective=aug_config.get('perspective', 0.0),
            flipud=aug_config.get('flipud', 0.0),
            fliplr=aug_config.get('fliplr', 0.5),
            mosaic=aug_config.get('mosaic', 1.0),
            mixup=aug_config.get('mixup', 0.0)
        )
        
        val_transforms = ValTransforms(img_size=img_size)
        
        # Select dataset class
        dataset_classes = {
            'coco': COCODataset,
            'voc': VOCDataset,
            'exdark': ExDarkDataset,
            'visdrone': VisDroneDataset,
            'fyp': FYPDataset
        }
        
        if dataset_name not in dataset_classes:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        DatasetClass = dataset_classes[dataset_name]
        
        # Create datasets
        print(f"Loading {dataset_name} dataset...")
        train_dataset = DatasetClass(
            root_dir=str(dataset_path),
            split='train',
            transform=train_transforms,
            target_size=img_size,
            augment=True
        )
        
        val_dataset = DatasetClass(
            root_dir=str(dataset_path),
            split='val',
            transform=val_transforms,
            target_size=img_size,
            augment=False
        )
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['train']['workers'],
            pin_memory=data_config['train']['pin_memory']
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=data_config['val']['batch_size'],
            shuffle=False,
            num_workers=data_config['val']['workers'],
            pin_memory=data_config['val']['pin_memory']
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        optimizer_config = self.config['train']['optimizer']
        scheduler_config = self.config['train']['scheduler']
        
        # Get optimizer parameters
        optimizer_name = optimizer_config.get('name', 'AdamW')
        lr = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0005)
        momentum = optimizer_config.get('momentum', 0.937)
        
        # Create optimizer
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(momentum, 0.999)
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=optimizer_config.get('nesterov', True)
            )
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Create scheduler
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        
        if scheduler_name.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['train']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_name.lower() == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=scheduler_config.get('patience', 10),
                factor=scheduler_config.get('factor', 0.1),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_name.lower() == 'multisteplr':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_config.get('milestones', [100, 200]),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def setup_loss(self):
        """Setup loss function"""
        loss_config = self.config['train']['loss']
        
        # Get number of classes
        if 'head' in self.config['model'] and 'num_classes' in self.config['model']['head']:
            num_classes = self.config['model']['head']['num_classes']
        else:
            dataset_name = self.config['data'].get('dataset', 'coco')
            num_classes = self.config['datasets'][dataset_name]['num_classes']
        
        # Create loss function
        criterion = YOLOSCEMALoss(
            num_classes=num_classes,
            box_gain=loss_config.get('box_gain', 7.5),
            cls_gain=loss_config.get('cls_gain', 0.5),
            obj_gain=loss_config.get('obj_gain', 1.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
        
        return criterion.to(self.device)
    
    def setup_ema(self):
        """Setup Exponential Moving Average"""
        ema_config = self.config['train']['tricks']['ema']
        decay = ema_config.get('decay', 0.9999)
        
        class EMA:
            def __init__(self, model, decay=0.9999):
                self.model = model
                self.decay = decay
                self.shadow = {}
                self.backup = {}
                
            def register(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
                
            def update(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        assert name in self.shadow
                        new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                        self.shadow[name] = new_average.clone()
                
            def apply_shadow(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        assert name in self.shadow
                        self.backup[name] = param.data
                        param.data = self.shadow[name]
                
            def restore(self):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        assert name in self.backup
                        param.data = self.backup[name]
                self.backup = {}
        
        ema = EMA(self.model, decay)
        ema.register()
        return ema
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['train']['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            
            # Prepare targets
            batch_targets = []
            for target in targets:
                batch_targets.append({
                    'boxes': target['boxes'].to(self.device),
                    'labels': target['labels'].to(self.device)
                })
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, batch_targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['train']['tricks']['gradient_clip_val'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['train']['tricks']['gradient_clip_val']
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions = self.model(images)
                loss = self.criterion(predictions, batch_targets)
                
                loss.backward()
                
                # Gradient clipping
                if self.config['train']['tricks']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['train']['tricks']['gradient_clip_val']
                    )
                
                self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log batch metrics
            self.global_step += 1
            self.logger.log_metrics({
                'train_loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=self.global_step, epoch=epoch)
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        # Apply EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        # Create evaluator
        evaluator = ModelEvaluator(
            self.model,
            device=self.device,
            iou_threshold=self.config['val']['metrics']['iou_threshold']
        )
        
        # Run evaluation
        print("Running validation...")
        metrics = evaluator.evaluate(
            self.val_loader,
            conf_threshold=self.config['val']['metrics']['conf_threshold'],
            nms_threshold=self.config['val']['metrics']['nms_threshold']
        )
        
        # Calculate validation loss
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        for images, targets in tqdm(self.val_loader, desc="Calculating validation loss"):
            images = images.to(self.device)
            
            # Prepare targets
            batch_targets = []
            for target in targets:
                batch_targets.append({
                    'boxes': target['boxes'].to(self.device),
                    'labels': target['labels'].to(self.device)
                })
            
            predictions = self.model(images)
            loss = self.criterion(predictions, batch_targets)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        
        # Restore original weights if EMA was applied
        if self.ema is not None:
            self.ema.restore()
        
        return avg_val_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        # Add EMA state if available
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        # Save checkpoint
        checkpoint_dir = self.exp_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        if self.config['train']['checkpoint']['max_checkpoints'] > 0:
            self.cleanup_checkpoints(checkpoint_dir)
    
    def cleanup_checkpoints(self, checkpoint_dir):
        """Remove old checkpoints"""
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > self.config['train']['checkpoint']['max_checkpoints']:
            for checkpoint in checkpoints[:-self.config['train']['checkpoint']['max_checkpoints']]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.metrics_history = checkpoint.get('metrics_history', {
            'train_loss': [],
            'val_loss': [],
            'val_mAP': []
        })
        
        # Load EMA state
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.shadow = checkpoint['ema_state_dict']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best metric: {self.best_metric:.4f}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting YOLO-SCEMA Training")
        print("="*50)
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Total epochs: {self.config['train']['epochs']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*50 + "\n")
        
        # Early stopping
        early_stopping = self.config['train'].get('early_stopping', {})
        early_stop_patience = early_stopping.get('patience', 50)
        early_stop_counter = 0
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['train']['epochs']):
            self.current_epoch = epoch + 1
            
            # Start epoch logging
            self.logger.start_epoch(epoch)
            
            # Train for one epoch
            train_loss = self.train_epoch(self.current_epoch)
            self.metrics_history['train_loss'].append(train_loss)
            
            # Validate
            if epoch % self.config['val']['frequency'] == 0:
                val_loss, val_metrics = self.validate(self.current_epoch)
                self.metrics_history['val_loss'].append(val_loss)
                self.metrics_history['val_mAP'].append(val_metrics.get('mAP', 0.0))
                
                # Log validation metrics
                self.logger.log_metrics({
                    'val_loss': val_loss,
                    'val_mAP': val_metrics.get('mAP', 0.0),
                    'val_mAP50': val_metrics.get('mAP@50', 0.0),
                    'val_mAP75': val_metrics.get('mAP@75', 0.0)
                }, step=self.global_step, epoch=self.current_epoch)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('mAP', 0.0))
                    else:
                        self.scheduler.step()
                
                # Save checkpoint if best
                current_metric = val_metrics.get('mAP', 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(self.current_epoch, is_best=True)
                    early_stop_counter = 0
                    print(f"New best model! mAP: {current_metric:.4f}")
                else:
                    early_stop_counter += 1
                
                # Save regular checkpoint
                if self.current_epoch % self.config['train']['checkpoint']['save_period'] == 0:
                    self.save_checkpoint(self.current_epoch)
                
                # Print epoch summary
                print(f"\nEpoch {self.current_epoch} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val mAP: {val_metrics.get('mAP', 0.0):.4f}")
                print(f"  Val mAP@50: {val_metrics.get('mAP@50', 0.0):.4f}")
                print(f"  Best mAP: {self.best_metric:.4f}")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # End epoch logging
            self.logger.end_epoch()
            
            # Early stopping check
            if early_stopping.get('enabled', False) and early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
                break
        
        # Save final model
        self.save_checkpoint(self.current_epoch)
        
        # Close logger
        self.logger.close()
        
        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Validation mAP: {self.best_metric:.4f}")
        print(f"Final model saved to: {self.exp_dir}")
        print("="*50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train YOLO-SCEMA model')
    parser.add_argument('--config', type=str, default='configs/yolov8_SCEMA.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        # Create default config directory if needed
        os.makedirs('configs', exist_ok=True)
        config_path = 'configs/yolov8_SCEMA.yaml'
    
    # Create trainer
    trainer = Trainer(config_path)
    
    # Override device if specified
    if args.device:
        trainer.config['train']['device'] = args.device
        trainer.device = trainer.setup_device()
        trainer.model = trainer.model.to(trainer.device)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()
