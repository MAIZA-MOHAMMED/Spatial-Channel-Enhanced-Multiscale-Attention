import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
import torch.nn.functional as F
import seaborn as sns


class DetectionVisualizer:
    """Visualization tools for object detection"""
    
    # Color palette for different classes
    COLOR_PALETTE = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 255, 128),  # Spring Green
        (255, 0, 128),  # Rose
    ]
    
    @staticmethod
    def draw_detections(image: np.ndarray,
                       boxes: np.ndarray,
                       scores: np.ndarray,
                       class_ids: np.ndarray,
                       class_names: List[str],
                       score_threshold: float = 0.3,
                       thickness: int = 2,
                       font_scale: float = 0.5) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image [H, W, C]
            boxes: Bounding boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            class_ids: Class indices [N]
            class_names: List of class names
            score_threshold: Minimum confidence score to display
            thickness: Line thickness
            font_scale: Font scale for labels
        
        Returns:
            Image with detections drawn
        """
        image = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score < score_threshold:
                continue
            
            # Get color for this class
            color = DetectionVisualizer.COLOR_PALETTE[class_id % len(DetectionVisualizer.COLOR_PALETTE)]
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label = f"{class_names[class_id]}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline),
                (x1 + text_width, y1),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return image
    
    @staticmethod
    def draw_heatmap(feature_map: torch.Tensor,
                    image: np.ndarray,
                    alpha: float = 0.5,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Draw feature heatmap on image
        
        Args:
            feature_map: Feature map [C, H, W] or [H, W]
            image: Input image [H, W, C]
            alpha: Transparency of heatmap
            colormap: OpenCV colormap
        
        Returns:
            Image with heatmap overlay
        """
        # Convert feature map to numpy
        if feature_map.dim() == 3:
            # Take mean over channels
            heatmap = feature_map.mean(dim=0).cpu().numpy()
        else:
            heatmap = feature_map.cpu().numpy()
        
        # Normalize heatmap to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        blended = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
        
        return blended
    
    @staticmethod
    def visualize_attention_maps(model: torch.nn.Module,
                                image: torch.Tensor,
                                layer_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Visualize attention maps from SCEMA modules
        
        Args:
            model: YOLO-SCEMA model
            image: Input image tensor [1, C, H, W]
            layer_names: Names of layers to visualize
        
        Returns:
            Dictionary of attention maps
        """
        attention_maps = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Take mean over channels for visualization
                attention_map = output.mean(dim=1, keepdim=True)
                attention_map = F.interpolate(
                    attention_map,
                    size=image.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                attention_maps[name] = attention_map.squeeze().detach().cpu().numpy()
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue
            
            if isinstance(module, (SCEMA, ChannelAttention, SpatialAttention, CrossSpatialLearning)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(image)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    @staticmethod
    def plot_detection_comparison(images: List[np.ndarray],
                                 detections_list: List[Dict],
                                 titles: List[str],
                                 class_names: List[str],
                                 score_threshold: float = 0.3):
        """
        Plot comparison of detections from multiple models
        
        Args:
            images: List of input images
            detections_list: List of detection dictionaries
            titles: Titles for each subplot
            class_names: List of class names
            score_threshold: Minimum confidence score to display
        """
        n_plots = len(images)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, image, detections, title in zip(axes, images, detections_list, titles):
            # Draw detections
            if 'boxes' in detections and len(detections['boxes']) > 0:
                boxes = detections['boxes'].cpu().numpy() if torch.is_tensor(detections['boxes']) else detections['boxes']
                scores = detections['scores'].cpu().numpy() if torch.is_tensor(detections['scores']) else detections['scores']
                class_ids = detections['classes'].cpu().numpy() if torch.is_tensor(detections['classes']) else detections['classes']
                
                image_with_detections = DetectionVisualizer.draw_detections(
                    image, boxes, scores, class_ids, class_names, score_threshold
                )
            else:
                image_with_detections = image
            
            ax.imshow(image_with_detections)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                               metric_name: str = 'mAP'):
        """
        Plot comparison of metrics from different models
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metric_name: Name of metric to plot
        """
        model_names = list(metrics_dict.keys())
        metric_values = [metrics_dict[name].get(metric_name, 0) for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(model_names)), metric_values)
        plt.xlabel('Models')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_loss_curves(train_losses: List[float],
                        val_losses: List[float] = None,
                        title: str = 'Training Loss'):
        """
        Plot training and validation loss curves
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            title: Plot title
        """
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses is not None:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add min loss annotations
        min_train_loss = min(train_losses)
        min_train_epoch = train_losses.index(min_train_loss) + 1
        plt.annotate(f'Min: {min_train_loss:.3f}', 
                    xy=(min_train_epoch, min_train_loss),
                    xytext=(min_train_epoch, min_train_loss * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
        
        if val_losses is not None:
            min_val_loss = min(val_losses)
            min_val_epoch = val_losses.index(min_val_loss) + 1
            plt.annotate(f'Min: {min_val_loss:.3f}', 
                        xy=(min_val_epoch, min_val_loss),
                        xytext=(min_val_epoch, min_val_loss * 1.1),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        horizontalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(precision: np.ndarray,
                                   recall: np.ndarray,
                                   ap: float,
                                   class_name: str = None):
        """
        Plot precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            ap: Average Precision
            class_name: Name of the class
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.fill_between(recall, precision, alpha=0.2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        
        title = f'Precision-Recall Curve'
        if class_name:
            title += f' ({class_name})'
        title += f'\nAP = {ap:.3f}'
        
        plt.title(title)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_grad_cam(model: torch.nn.Module,
                       image: torch.Tensor,
                       target_layer: torch.nn.Module,
                       target_class: int = None):
        """
        Create Grad-CAM visualization
        
        Args:
            model: YOLO-SCEMA model
            image: Input image tensor [1, C, H, W]
            target_layer: Target layer for Grad-CAM
            target_class: Target class index (None for highest scoring class)
        
        Returns:
            Grad-CAM heatmap
        """
        model.eval()
        
        # Forward hook to store features
        features = []
        def forward_hook(module, input, output):
            features.append(output)
        
        # Backward hook to store gradients
        gradients = []
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        predictions = model(image)
        
        # Get target class
        if target_class is None:
            # Use highest scoring class
            cls_outputs, _ = predictions
            scores = torch.sigmoid(cls_outputs[-1]).max(dim=1)[0]  # Last scale
            target_class = torch.argmax(scores.mean(dim=(1, 2)))
        
        # Zero gradients
        model.zero_grad()
        
        # Create one-hot target
        target = torch.zeros_like(cls_outputs[-1])
        target[:, target_class, :, :] = 1
        
        # Backward pass
        cls_outputs[-1].backward(gradient=target, retain_graph=True)
        
        # Get features and gradients
        feature_maps = features[0]
        grads = gradients[0]
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        
        # Weight feature maps by gradients
        for i in range(feature_maps.size(1)):
            feature_maps[:, i, :, :] *= pooled_grads[i]
        
        # Apply ReLU and average over channels
        cam = torch.relu(torch.sum(feature_maps, dim=1))
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return cam.squeeze().detach().cpu().numpy()
