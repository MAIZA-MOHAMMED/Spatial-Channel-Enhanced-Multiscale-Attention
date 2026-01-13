import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger:
    """Unified logger for training and evaluation"""
    def __init__(self, 
                 log_dir: str = 'logs',
                 experiment_name: str = None,
                 use_tensorboard: bool = True,
                 use_csv: bool = True,
                 use_json: bool = True):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
            use_tensorboard: Whether to use TensorBoard
            use_csv: Whether to save CSV logs
            use_json: Whether to save JSON logs
        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize loggers
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.use_json = use_json
        
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        
        if use_csv:
            self.csv_file = os.path.join(self.log_dir, 'metrics.csv')
            self._init_csv()
        
        if use_json:
            self.json_file = os.path.join(self.log_dir, 'config.json')
            self.metrics_history = {}
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.start_time = time.time()
        
        # Buffer for current epoch metrics
        self.current_metrics = {}
        
        print(f"Logger initialized. Logs will be saved to: {self.log_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with header"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'global_step', 'timestamp', 'metric', 'value'])
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        config_copy = config.copy()
        
        # Convert non-serializable objects to strings
        for key, value in config_copy.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_copy[key] = str(value)
        
        # Save to JSON
        if self.use_json:
            with open(self.json_file, 'w') as f:
                json.dump(config_copy, f, indent=2)
        
        # Save to text file
        config_txt = os.path.join(self.log_dir, 'config.txt')
        with open(config_txt, 'w') as f:
            for key, value in config_copy.items():
                f.write(f"{key}: {value}\n")
        
        print("Configuration logged.")
    
    def log_metrics(self, 
                   metrics: Dict[str, Union[float, int]],
                   step: int = None,
                   prefix: str = '',
                   epoch: int = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (default: global_step)
            prefix: Prefix for metric names
            epoch: Epoch number
        """
        if step is None:
            step = self.global_step
        
        if epoch is None:
            epoch = self.epoch
        
        timestamp = time.time() - self.start_time
        
        # Add prefix to metric names
        full_metrics = {}
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            full_metrics[full_key] = value
        
        # Update current epoch metrics
        for key, value in full_metrics.items():
            if key not in self.current_metrics:
                self.current_metrics[key] = []
            self.current_metrics[key].append(value)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in full_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Log to CSV
        if self.use_csv:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for key, value in full_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.writerow([epoch, step, timestamp, key, value])
        
        # Log to JSON history
        if self.use_json:
            for key, value in full_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append({
                    'epoch': epoch,
                    'step': step,
                    'timestamp': timestamp,
                    'value': value
                })
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard"""
        if self.use_tensorboard:
            try:
                self.tb_writer.add_graph(model, input_tensor)
                print("Model graph logged to TensorBoard.")
            except Exception as e:
                print(f"Failed to log model graph: {e}")
    
    def log_histograms(self, model: torch.nn.Module, step: int = None):
        """Log weight histograms to TensorBoard"""
        if step is None:
            step = self.global_step
        
        if self.use_tensorboard:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.tb_writer.add_histogram(name, param.data.cpu().numpy(), step)
    
    def log_images(self, 
                  tag: str,
                  images: torch.Tensor,
                  step: int = None,
                  dataformats: str = 'NCHW'):
        """
        Log images to TensorBoard
        
        Args:
            tag: Tag for the images
            images: Tensor of images [N, C, H, W] or [N, H, W, C]
            step: Step number
            dataformats: Format of images ('NCHW' or 'NHWC')
        """
        if step is None:
            step = self.global_step
        
        if self.use_tensorboard:
            self.tb_writer.add_images(tag, images, step, dataformats=dataformats)
    
    def log_attention_maps(self, 
                          tag: str,
                          attention_maps: Dict[str, np.ndarray],
                          step: int = None):
        """
        Log attention maps to TensorBoard
        
        Args:
            tag: Tag prefix for attention maps
            attention_maps: Dictionary of attention maps
            step: Step number
        """
        if step is None:
            step = self.global_step
        
        if self.use_tensorboard:
            for name, attention_map in attention_maps.items():
                # Normalize attention map
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-7)
                
                # Convert to 3-channel for visualization
                if attention_map.ndim == 2:
                    attention_map = np.stack([attention_map] * 3, axis=0)
                elif attention_map.ndim == 3 and attention_map.shape[0] == 1:
                    attention_map = np.repeat(attention_map, 3, axis=0)
                
                # Add to TensorBoard
                self.tb_writer.add_image(f"{tag}/{name}", attention_map, step)
    
    def start_epoch(self, epoch: int):
        """Start a new epoch"""
        self.epoch = epoch
        self.current_metrics = {}
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}")
        print(f"{'='*50}")
    
    def end_epoch(self):
        """End current epoch and log epoch summary"""
        epoch_summary = {}
        
        # Calculate average metrics for the epoch
        for key, values in self.current_metrics.items():
            if values:
                epoch_summary[f"{key}_mean"] = np.mean(values)
                epoch_summary[f"{key}_std"] = np.std(values)
                epoch_summary[f"{key}_min"] = np.min(values)
                epoch_summary[f"{key}_max"] = np.max(values)
        
        # Log epoch summary
        self.log_metrics(epoch_summary, step=self.global_step, prefix='epoch')
        
        # Print epoch summary
        print(f"\nEpoch {self.epoch} Summary:")
        for key in sorted(epoch_summary.keys()):
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '')
                mean_val = epoch_summary[key]
                std_val = epoch_summary.get(f"{metric_name}_std", 0)
                print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save checkpoint
        self.save_checkpoint()
    
    def update_global_step(self, increment: int = 1):
        """Update global step counter"""
        self.global_step += increment
    
    def save_checkpoint(self, 
                       model: torch.nn.Module = None,
                       optimizer: torch.optim.Optimizer = None,
                       scheduler: Any = None,
                       filename: str = None):
        """
        Save training checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            filename: Checkpoint filename (default: checkpoint_epoch_{epoch}.pth)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pth"
        
        checkpoint_path = os.path.join(self.log_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'timestamp': datetime.now().isoformat(),
        }
        
        if model is not None:
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['model_config'] = getattr(model, 'config', {})
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save metrics history
        if self.use_json:
            checkpoint['metrics_history'] = self.metrics_history
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module = None,
                       optimizer: torch.optim.Optimizer = None,
                       scheduler: Any = None) -> Dict[str, Any]:
        """
        Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state dict into
            optimizer: Optimizer to load state dict into
            scheduler: Scheduler to load state dict into
        
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Update logger state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        # Load model state
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from checkpoint (epoch {self.epoch})")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded from checkpoint")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded from checkpoint")
        
        # Load metrics history
        if self.use_json and 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        
        return checkpoint
    
    def log_progress(self, 
                    current: int,
                    total: int,
                    prefix: str = '',
                    suffix: str = '',
                    decimals: int = 1,
                    length: int = 50,
                    fill: str = '█'):
        """
        Print progress bar
        
        Args:
            current: Current progress
            total: Total progress
            prefix: Prefix string
            suffix: Suffix string
            decimals: Number of decimal places in percentage
            length: Length of progress bar
            fill: Character to fill progress bar
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        
        # Print new line when complete
        if current == total:
            print()
    
    def close(self):
        """Close logger and clean up"""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        # Save final metrics history
        if self.use_json:
            history_file = os.path.join(self.log_dir, 'metrics_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Logs saved to: {self.log_dir}")
        print(f"{'='*50}")


class ProgressLogger:
    """Simple progress logger for training loops"""
    def __init__(self, total_steps: int, update_freq: int = 10):
        """
        Initialize progress logger
        
        Args:
            total_steps: Total number of steps
            update_freq: Frequency of updates (steps)
        """
        self.total_steps = total_steps
        self.update_freq = update_freq
        self.current_step = 0
        self.start_time = time.time()
        self.losses = []
    
    def update(self, loss: float, metrics: Dict[str, float] = None):
        """
        Update progress
        
        Args:
            loss: Current loss value
            metrics: Additional metrics to log
        """
        self.current_step += 1
        self.losses.append(loss)
        
        if self.current_step % self.update_freq == 0 or self.current_step == self.total_steps:
            avg_loss = np.mean(self.losses[-self.update_freq:])
            elapsed = time.time() - self.start_time
            steps_per_sec = self.current_step / elapsed
            
            progress = f"Step {self.current_step}/{self.total_steps} | "
            progress += f"Loss: {avg_loss:.4f} | "
            progress += f"Speed: {steps_per_sec:.2f} steps/s"
            
            if metrics:
                for key, value in metrics.items():
                    progress += f" | {key}: {value:.4f}"
            
            print(progress)
    
    def reset(self):
        """Reset logger"""
        self.current_step = 0
        self.start_time = time.time()
        self.losses = []
