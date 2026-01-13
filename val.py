"""
YOLO-SCEMA Validation Script

Validation script for evaluating YOLO-SCEMA model performance.
Computes metrics like mAP, precision, recall, and visualizes results.

Author: Mohammed MAIZA
Repository: https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import create_model
from data import create_dataloader, COCODataset, VOCDataset, ExDarkDataset, VisDroneDataset, FYPDataset
from data.transforms import ValTransforms
from utils.metrics import ModelEvaluator, DetectionMetrics
from utils.visualizer import DetectionVisualizer
from utils.logger import Logger

class Validator:
    """YOLO-SCEMA Validator"""
    
    def __init__(self, config_path, weights_path=None):
        """
        Initialize validator
        
        Args:
            config_path: Path to configuration file
            weights_path: Path to model weights
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup device
        self.device = self.setup_device()
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(weights_path)
        
        # Setup validation dataset
        self.val_loader = self.setup_validation_dataset()
        
        # Get class names
        self.class_names = self.get_class_names()
        
        # Setup logger
        self.logger = self.setup_logger()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup validation device"""
        device = self.config['train'].get('device', 'cuda')
        
        if device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def load_model(self, weights_path):
        """Load YOLO-SCEMA model"""
        model_config = self.config['model']
        
        # Get model size
        model_size = model_config.get('size', 'n')
        
        # Get number of classes
        if 'head' in model_config and 'num_classes' in model_config['head']:
            num_classes = model_config['head']['num_classes']
        else:
            # Get from dataset config
            dataset_name = self.config['data'].get('dataset', 'coco')
            num_classes = self.config['datasets'][dataset_name]['num_classes']
        
        # Create model
        model = create_model(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        if weights_path is None:
            # Try to find weights in config
            weights_path = self.config['val'].get('weights', None)
        
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Loaded weights from {weights_path}")
        else:
            raise ValueError(f"Weights file not found: {weights_path}")
        
        # Move to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        print(f"Model: YOLO-SCEMA-{model_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Input size: {self.config['data']['img_size']}")
        
        return model
    
    def setup_validation_dataset(self):
        """Setup validation dataset and dataloader"""
        data_config = self.config['data']
        
        # Get dataset name
        dataset_name = data_config.get('dataset', 'coco')
        dataset_path = Path(data_config['dataset_path'])
        
        # Get image size
        img_size = data_config.get('img_size', [640, 640])
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        # Create transforms
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
        
        # Create validation dataset
        print(f"Loading {dataset_name} validation dataset...")
        val_dataset = DatasetClass(
            root_dir=str(dataset_path),
            split='val',
            transform=val_transforms,
            target_size=img_size,
            augment=False
        )
        
        # Create dataloader
        val_loader = create_dataloader(
            val_dataset,
            batch_size=data_config['val']['batch_size'],
            shuffle=False,
            num_workers=data_config['val']['workers'],
            pin_memory=data_config['val']['pin_memory']
        )
        
        print(f"Validation samples: {len(val_dataset)}")
        
        return val_loader
    
    def get_class_names(self):
        """Get class names for the current dataset"""
        dataset_name = self.config['data'].get('dataset', 'coco')
        
        if dataset_name in self.config['datasets']:
            return self.config['datasets'][dataset_name]['class_names']
        else:
            # Default COCO classes
            return [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
    
    def setup_logger(self):
        """Setup logger for validation results"""
        # Create validation results directory
        results_dir = Path(self.config['val']['visualization']['save_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = Logger(
            log_dir=str(results_dir),
            experiment_name='validation',
            use_tensorboard=False,
            use_csv=True,
            use_json=True
        )
        
        return logger
    
    def evaluate(self, visualize=False):
        """
        Evaluate model on validation set
        
        Args:
            visualize: Whether to visualize predictions
        
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*50)
        print("Starting YOLO-SCEMA Validation")
        print("="*50)
        
        # Create evaluator
        evaluator = ModelEvaluator(
            self.model,
            device=self.device,
            iou_threshold=self.config['val']['metrics']['iou_threshold']
        )
        
        # Run evaluation
        print("Evaluating model...")
        metrics = evaluator.evaluate(
            self.val_loader,
            conf_threshold=self.config['val']['metrics']['conf_threshold'],
            nms_threshold=self.config['val']['metrics']['nms_threshold']
        )
        
        # Print metrics
        print("\n" + "="*50)
        print("Validation Results")
        print("="*50)
        
        # Print overall metrics
        print(f"\nOverall Metrics:")
        print(f"  mAP:      {metrics.get('mAP', 0.0):.4f}")
        print(f"  mAP@50:   {metrics.get('mAP@50', 0.0):.4f}")
        print(f"  mAP@75:   {metrics.get('mAP@75', 0.0):.4f}")
        print(f"  mAP@95:   {metrics.get('mAP@95', 0.0):.4f}")
        
        # Print per-class AP if available
        class_aps = [(i, metrics.get(f'AP_{i}', 0.0)) for i in range(len(self.class_names))]
        class_aps.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Classes by AP:")
        for i, (class_idx, ap) in enumerate(class_aps[:10]):
            class_name = self.class_names[class_idx]
            print(f"  {i+1:2d}. {class_name:20s}: {ap:.4f}")
        
        print(f"\nBottom 10 Classes by AP:")
        for i, (class_idx, ap) in enumerate(class_aps[-10:]):
            class_name = self.class_names[class_idx]
            print(f"  {i+1:2d}. {class_name:20s}: {ap:.4f}")
        
        # Log metrics
        self.logger.log_metrics(metrics, step=0, prefix='val')
        
        # Save metrics to file
        metrics_file = Path(self.config['val']['visualization']['save_dir']) / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Visualize predictions if requested
        if visualize:
            self.visualize_predictions()
        
        return metrics
    
    def visualize_predictions(self):
        """Visualize model predictions on validation set"""
        print("\nGenerating visualizations...")
        
        # Create visualization directory
        vis_dir = Path(self.config['val']['visualization']['save_dir']) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get maximum number of images to visualize
        max_images = self.config['val']['visualization']['max_images']
        
        # Process images
        self.model.eval()
        image_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc="Visualizing")):
                # Stop if we've reached the maximum
                if image_count >= max_images:
                    break
                
                # Move images to device
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Decode predictions for each image in batch
                for i in range(len(images)):
                    if image_count >= max_images:
                        break
                    
                    # Get image and target
                    image_tensor = images[i]
                    target = targets[i]
                    
                    # Convert image tensor to numpy
                    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
                    image_np = (image_np * 255).astype(np.uint8)
                    
                    # Convert BGR to RGB for visualization
                    image_rgb = image_np[:, :, ::-1].copy()
                    
                    # Get predictions for this image
                    cls_outputs = [pred[i:i+1] for pred in predictions[0]]
                    reg_outputs = [pred[i:i+1] for pred in predictions[1]]
                    
                    # Decode predictions
                    detections = self.decode_predictions(
                        (cls_outputs, reg_outputs),
                        orig_shape=image_rgb.shape[:2]
                    )
                    
                    # Get ground truth
                    gt_boxes = target['boxes'].cpu().numpy()
                    gt_labels = target['labels'].cpu().numpy()
                    
                    # Create visualization
                    fig = self.create_comparison_visualization(
                        image_rgb,
                        detections,
                        gt_boxes,
                        gt_labels
                    )
                    
                    # Save visualization
                    vis_path = vis_dir / f"pred_{image_count:04d}.png"
                    fig.savefig(vis_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    
                    image_count += 1
        
        print(f"Visualizations saved to: {vis_dir}")
    
    def decode_predictions(self, predictions, orig_shape):
        """Decode model predictions (similar to detect.py but simplified)"""
        cls_outputs, reg_outputs = predictions
        
        detections = {
            'boxes': [],
            'scores': [],
            'classes': []
        }
        
        # Process each scale
        for scale_idx, (cls_pred, reg_pred) in enumerate(zip(cls_outputs, reg_outputs)):
            stride = 8 * (2 ** scale_idx)
            grid_h, grid_w = cls_pred.shape[2:]
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, device=self.device),
                torch.arange(grid_w, device=self.device)
            )
            grid = torch.stack([grid_x, grid_y], dim=0).float()
            
            # Decode boxes
            boxes = self.decode_boxes(reg_pred[0], grid, stride)
            
            # Get class predictions
            cls_scores, cls_indices = torch.max(
                torch.sigmoid(cls_pred[0]), dim=0
            )
            
            # Objectness score
            obj_score = torch.sigmoid(
                (reg_pred[0][0] + reg_pred[0][1]) / 2
            )
            
            # Combine scores
            scores = cls_scores * obj_score
            
            # Filter by confidence
            mask = scores > self.config['val']['metrics']['conf_threshold']
            if not mask.any():
                continue
            
            boxes = boxes.view(-1, 4)[mask.view(-1)]
            scores = scores[mask]
            classes = cls_indices[mask]
            
            # Apply NMS
            keep = self.nms(boxes, scores, self.config['val']['metrics']['nms_threshold'])
            
            detections['boxes'].append(boxes[keep])
            detections['scores'].append(scores[keep])
            detections['classes'].append(classes[keep])
        
        # Merge detections
        if detections['boxes']:
            boxes = torch.cat(detections['boxes'], dim=0).cpu().numpy()
            scores = torch.cat(detections['scores'], dim=0).cpu().numpy()
            classes = torch.cat(detections['classes'], dim=0).cpu().numpy()
        else:
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        }
    
    def decode_boxes(self, reg_pred, grid, stride):
        """Decode regression predictions to boxes"""
        center_x = (reg_pred[0].sigmoid() + grid[0]) * stride
        center_y = (reg_pred[1].sigmoid() + grid[1]) * stride
        width = torch.exp(reg_pred[2]) * stride
        height = torch.exp(reg_pred[3]) * stride
        
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=0)
        boxes = boxes.permute(1, 2, 0).reshape(-1, 4)
        
        return boxes
    
    def nms(self, boxes, scores, threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        sorted_scores, indices = torch.sort(scores, descending=True)
        boxes = boxes[indices]
        
        keep = []
        while len(boxes) > 0:
            keep.append(indices[0].item())
            
            if len(boxes) == 1:
                break
            
            ious = self.box_iou(boxes[0:1], boxes[1:]).squeeze(0)
            mask = ious < threshold
            boxes = boxes[1:][mask]
            indices = indices[1:][mask]
            sorted_scores = sorted_scores[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def box_iou(self, box1, box2):
        """Calculate IoU between two sets of boxes"""
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def create_comparison_visualization(self, image, detections, gt_boxes, gt_labels):
        """Create comparison visualization between predictions and ground truth"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(image)
        for box, label in zip(gt_boxes, gt_labels):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                axes[1].add_patch(rect)
                
                if label < len(self.class_names):
                    class_name = self.class_names[int(label)]
                    axes[1].text(
                        x1, y1 - 5, class_name,
                        color='green', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predictions
        axes[2].imshow(image)
        for box, score, class_idx in zip(detections['boxes'], detections['scores'], detections['classes']):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[2].add_patch(rect)
                
                if class_idx < len(self.class_names):
                    class_name = self.class_names[int(class_idx)]
                    axes[2].text(
                        x1, y1 - 5, f"{class_name}: {score:.2f}",
                        color='red', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
        axes[2].set_title('Predictions')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate YOLO-SCEMA model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--config', type=str, default='configs/yolov8_SCEMA.yaml',
                       help='Path to configuration file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Create validator
    validator = Validator(args.config, args.weights)
    
    # Override device if specified
    if args.device:
        validator.config['train']['device'] = args.device
        validator.device = validator.setup_device()
        validator.model = validator.model.to(validator.device)
    
    # Run evaluation
    metrics = validator.evaluate(visualize=args.visualize)
    
    # Save results summary
    results_summary = {
        'model': validator.config['model'].get('size', 'n'),
        'dataset': validator.config['data'].get('dataset', 'coco'),
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_file = Path(validator.config['val']['visualization']['save_dir']) / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults summary saved to: {summary_file}")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
