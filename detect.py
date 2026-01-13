"""
YOLO-SCEMA Inference Script

Inference script for the YOLO-SCEMA model.
Supports image, video, and webcam inference with various visualization options.

Author: Mohammed MAIZA
Repository: https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention
"""

import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import create_model
from data.transforms import TestTransforms
from utils.visualizer import DetectionVisualizer

class Detector:
    """YOLO-SCEMA Detector"""
    
    def __init__(self, config_path: str, weights_path: str = None):
        """
        Initialize detector
        
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
        
        # Setup transforms
        self.transforms = TestTransforms(
            img_size=self.config['data']['img_size'],
            stride=32
        )
        
        # Get class names
        self.class_names = self.get_class_names()
        
        # Inference parameters
        self.conf_threshold = self.config['inference']['conf_threshold']
        self.iou_threshold = self.config['inference']['iou_threshold']
        self.max_detections = self.config['inference']['max_detections']
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup inference device"""
        device = self.config['train'].get('device', 'cuda')
        
        if device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def load_model(self, weights_path: Optional[str] = None):
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
            weights_path = self.config['inference'].get('weights', None)
        
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
            print("Warning: No weights loaded. Using randomly initialized model.")
        
        # Move to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        print(f"Model: YOLO-SCEMA-{model_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Input size: {self.config['data']['img_size']}")
        
        return model
    
    def get_class_names(self) -> List[str]:
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
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image in BGR format (OpenCV default)
        
        Returns:
            Preprocessed tensor and metadata
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor, metadata = self.transforms(image_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor, metadata
    
    def postprocess_detections(
        self,
        predictions: Tuple[List[torch.Tensor], List[torch.Tensor]],
        metadata: Dict[str, Any],
        orig_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Postprocess model predictions
        
        Args:
            predictions: Model predictions (cls_outputs, reg_outputs)
            metadata: Preprocessing metadata
            orig_shape: Original image shape (height, width)
        
        Returns:
            Processed detections
        """
        cls_outputs, reg_outputs = predictions
        batch_size = cls_outputs[0].shape[0]
        
        detections = {
            'boxes': [],
            'scores': [],
            'classes': []
        }
        
        # Process each scale
        for scale_idx, (cls_pred, reg_pred) in enumerate(zip(cls_outputs, reg_outputs)):
            stride = 8 * (2 ** scale_idx)
            grid_h, grid_w = cls_pred.shape[2:]
            
            # Process each image in batch (usually batch_size=1 for inference)
            for batch_idx in range(batch_size):
                # Get predictions for this batch
                cls_pred_scale = cls_pred[batch_idx]  # [C, H, W]
                reg_pred_scale = reg_pred[batch_idx]  # [4, H, W]
                
                # Create grid
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(grid_h, device=self.device),
                    torch.arange(grid_w, device=self.device)
                )
                grid = torch.stack([grid_x, grid_y], dim=0).float()
                
                # Decode boxes
                boxes = self.decode_boxes(reg_pred_scale, grid, stride)
                
                # Get class predictions
                cls_scores, cls_indices = torch.max(
                    torch.sigmoid(cls_pred_scale), dim=0
                )
                
                # Objectness score (simplified)
                obj_score = torch.sigmoid(
                    (reg_pred_scale[0] + reg_pred_scale[1]) / 2
                )
                
                # Combine scores
                scores = cls_scores * obj_score
                
                # Filter by confidence
                mask = scores > self.conf_threshold
                if not mask.any():
                    continue
                
                boxes = boxes.view(-1, 4)[mask.view(-1)]
                scores = scores[mask]
                classes = cls_indices[mask]
                
                # Apply NMS
                keep = self.nms(boxes, scores, self.iou_threshold)
                
                # Limit number of detections
                if len(keep) > self.max_detections:
                    keep = keep[:self.max_detections]
                
                detections['boxes'].append(boxes[keep])
                detections['scores'].append(scores[keep])
                detections['classes'].append(classes[keep])
        
        # Merge detections from all scales
        if detections['boxes']:
            boxes = torch.cat(detections['boxes'], dim=0)
            scores = torch.cat(detections['scores'], dim=0)
            classes = torch.cat(detections['classes'], dim=0)
            
            # Apply scale and padding to convert back to original coordinates
            scale = metadata['scale']
            pad_top, pad_right, pad_bottom, pad_left = metadata['pad']
            
            # Remove padding and apply scaling
            boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
            boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
            boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
            boxes[:, 3] = (boxes[:, 3] - pad_top) / scale
            
            # Clip to image bounds
            boxes[:, 0] = boxes[:, 0].clamp(0, orig_shape[1])
            boxes[:, 1] = boxes[:, 1].clamp(0, orig_shape[0])
            boxes[:, 2] = boxes[:, 2].clamp(0, orig_shape[1])
            boxes[:, 3] = boxes[:, 3].clamp(0, orig_shape[0])
        else:
            boxes = torch.tensor([], device=self.device)
            scores = torch.tensor([], device=self.device)
            classes = torch.tensor([], device=self.device, dtype=torch.long)
        
        return {
            'boxes': boxes.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'classes': classes.cpu().numpy()
        }
    
    def decode_boxes(self, reg_pred: torch.Tensor, grid: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Decode regression predictions to bounding boxes
        
        Args:
            reg_pred: Regression predictions [4, H, W]
            grid: Grid coordinates [2, H, W]
            stride: Feature map stride
        
        Returns:
            Decoded boxes in xyxy format
        """
        # Center coordinates
        center_x = (reg_pred[0].sigmoid() + grid[0]) * stride
        center_y = (reg_pred[1].sigmoid() + grid[1]) * stride
        
        # Width and height
        width = torch.exp(reg_pred[2]) * stride
        height = torch.exp(reg_pred[3]) * stride
        
        # Convert to xyxy format
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=0)
        boxes = boxes.permute(1, 2, 0).reshape(-1, 4)
        
        return boxes
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Bounding boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            threshold: IoU threshold
        
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Sort by scores (descending)
        sorted_scores, indices = torch.sort(scores, descending=True)
        boxes = boxes[indices]
        
        keep = []
        while len(boxes) > 0:
            # Keep the box with highest score
            keep.append(indices[0].item())
            
            if len(boxes) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = self.box_iou(boxes[0:1], boxes[1:]).squeeze(0)
            
            # Remove boxes with high IoU
            mask = ious < threshold
            boxes = boxes[1:][mask]
            indices = indices[1:][mask]
            sorted_scores = sorted_scores[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        # Calculate intersection
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def detect_image(self, image_path: str, save: bool = True) -> Dict[str, Any]:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            save: Whether to save visualization
        
        Returns:
            Detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Store original image
        orig_image = image.copy()
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        start_time = time.time()
        tensor, metadata = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(tensor)
            inference_time = time.time() - start_time
        
        # Postprocess
        start_time = time.time()
        detections = self.postprocess_detections(predictions, metadata, (orig_h, orig_w))
        postprocess_time = time.time() - start_time
        
        # Print timing
        print(f"\nImage: {image_path}")
        print(f"  Original size: {orig_w}x{orig_h}")
        print(f"  Preprocess: {preprocess_time*1000:.1f}ms")
        print(f"  Inference: {inference_time*1000:.1f}ms")
        print(f"  Postprocess: {postprocess_time*1000:.1f}ms")
        print(f"  Total: {(preprocess_time+inference_time+postprocess_time)*1000:.1f}ms")
        print(f"  Detections: {len(detections['boxes'])}")
        
        # Visualize
        if self.config['inference']['visualization']['enabled']:
            # Convert boxes to integers for visualization
            boxes_int = detections['boxes'].astype(int)
            
            # Draw detections
            visualized = DetectionVisualizer.draw_detections(
                orig_image,
                boxes_int,
                detections['scores'],
                detections['classes'].astype(int),
                self.class_names,
                score_threshold=self.conf_threshold,
                thickness=self.config['inference']['visualization']['thickness'],
                font_scale=self.config['inference']['visualization']['font_scale']
            )
            
            # Save visualization
            if save:
                output_dir = Path(self.config['inference']['output']['save_dir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create output filename
                input_path = Path(image_path)
                output_path = output_dir / f"{input_path.stem}_detected{input_path.suffix}"
                
                # Save image
                cv2.imwrite(str(output_path), visualized)
                print(f"  Saved to: {output_path}")
            
            # Show image if in interactive mode
            if self.config['inference']['visualization'].get('show', False):
                cv2.imshow('Detection', visualized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return {
            'image_path': image_path,
            'detections': detections,
            'timing': {
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
        }
    
    def detect_video(self, video_path: str, save: bool = True):
        """
        Detect objects in a video
        
        Args:
            video_path: Path to input video
            save: Whether to save output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        
        # Create video writer if saving
        if save:
            output_dir = Path(self.config['inference']['output']['save_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            input_path = Path(video_path)
            output_path = output_dir / f"{input_path.stem}_detected{input_path.suffix}"
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_time = 0.0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            start_time = time.time()
            
            # Preprocess
            tensor, metadata = self.preprocess_image(frame)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(tensor)
            
            # Postprocess
            detections = self.postprocess_detections(predictions, metadata, (height, width))
            
            # Visualize
            if self.config['inference']['visualization']['enabled']:
                boxes_int = detections['boxes'].astype(int)
                visualized = DetectionVisualizer.draw_detections(
                    frame,
                    boxes_int,
                    detections['scores'],
                    detections['classes'].astype(int),
                    self.class_names,
                    score_threshold=self.conf_threshold
                )
            else:
                visualized = frame
            
            # Write frame
            if save:
                out.write(visualized)
            
            # Update statistics
            frame_time = time.time() - start_time
            total_time += frame_time
            frame_count += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'fps': f"{1.0/frame_time:.1f}",
                'detections': len(detections['boxes'])
            })
        
        # Cleanup
        pbar.close()
        cap.release()
        if save:
            out.release()
            print(f"\nSaved video to: {output_path}")
        
        # Print statistics
        avg_fps = frame_count / total_time
        print(f"\nVideo processing complete:")
        print(f"  Processed frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
    
    def detect_webcam(self, camera_id: int = 0):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nWebcam: Camera {camera_id}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print("  Press 'q' to quit")
        
        # FPS calculation
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            tensor, metadata = self.preprocess_image(frame)
            
            with torch.no_grad():
                predictions = self.model(tensor)
            
            detections = self.postprocess_detections(predictions, metadata, (height, width))
            
            # Visualize
            boxes_int = detections['boxes'].astype(int)
            visualized = DetectionVisualizer.draw_detections(
                frame,
                boxes_int,
                detections['scores'],
                detections['classes'].astype(int),
                self.class_names,
                score_threshold=self.conf_threshold
            )
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            
            # Display FPS
            cv2.putText(
                visualized,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display detections count
            cv2.putText(
                visualized,
                f"Detections: {len(detections['boxes'])}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('YOLO-SCEMA Webcam', visualized)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\nWebcam processing complete:")
        print(f"  Processed frames: {frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
    
    def detect_directory(self, directory_path: str):
        """
        Detect objects in all images in a directory
        
        Args:
            directory_path: Path to directory containing images
        """
        # Get image files
        directory = Path(directory_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return
        
        print(f"\nFound {len(image_files)} images in {directory_path}")
        
        # Process each image
        results = []
        total_time = 0.0
        
        for image_file in tqdm(image_files, desc="Processing images"):
            result = self.detect_image(str(image_file), save=True)
            results.append(result)
            total_time += result['timing']['total']
        
        # Print summary
        print(f"\nDirectory processing complete:")
        print(f"  Processed images: {len(results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {total_time/len(results):.2f}s")
        print(f"  Results saved to: {self.config['inference']['output']['save_dir']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLO-SCEMA Inference')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image path, video path, directory path, or "webcam"')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--config', type=str, default='configs/yolov8_SCEMA.yaml',
                       help='Path to configuration file')
    parser.add_argument('--conf-threshold', type=float, default=None,
                       help='Confidence threshold (overrides config)')
    parser.add_argument('--iou-threshold', type=float, default=None,
                       help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output')
    parser.add_argument('--show', action='store_true',
                       help='Show visualization')
    
    args = parser.parse_args()
    
    # Load detector
    detector = Detector(args.config, args.weights)
    
    # Override thresholds if specified
    if args.conf_threshold is not None:
        detector.conf_threshold = args.conf_threshold
        detector.config['inference']['conf_threshold'] = args.conf_threshold
    
    if args.iou_threshold is not None:
        detector.iou_threshold = args.iou_threshold
        detector.config['inference']['iou_threshold'] = args.iou_threshold
    
    # Override device if specified
    if args.device:
        detector.config['train']['device'] = args.device
        detector.device = detector.setup_device()
        detector.model = detector.model.to(detector.device)
    
    # Override show setting
    if args.show:
        detector.config['inference']['visualization']['show'] = True
    
    # Determine source type and process
    source = args.source.lower()
    save = not args.no_save
    
    if source == 'webcam':
        detector.detect_webcam()
    elif os.path.isdir(args.source):
        detector.detect_directory(args.source)
    elif os.path.isfile(args.source):
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if Path(args.source).suffix.lower() in video_extensions:
            detector.detect_video(args.source, save=save)
        else:
            # Assume it's an image
            detector.detect_image(args.source, save=save)
    else:
        print(f"Invalid source: {args.source}")
        print("Supported sources: image file, video file, directory, or 'webcam'")

if __name__ == '__main__':
    main()
