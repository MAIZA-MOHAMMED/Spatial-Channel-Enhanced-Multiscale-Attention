import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from collections import defaultdict


class DetectionMetrics:
    """Metrics calculator for object detection"""
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = defaultdict(list)  # True positives per class
        self.fp = defaultdict(list)  # False positives per class
        self.fn = defaultdict(list)  # False negatives per class
        self.scores = defaultdict(list)  # Confidence scores per class
        self.gt_counts = defaultdict(int)  # Ground truth counts per class
        
    def update(self, 
               pred_boxes: List[torch.Tensor],
               pred_scores: List[torch.Tensor],
               pred_classes: List[torch.Tensor],
               gt_boxes: List[torch.Tensor],
               gt_classes: List[torch.Tensor]):
        """
        Update metrics with a batch of predictions
        
        Args:
            pred_boxes: List of predicted bounding boxes [N, 4] in xyxy format
            pred_scores: List of confidence scores [N]
            pred_classes: List of predicted class indices [N]
            gt_boxes: List of ground truth boxes [M, 4] in xyxy format
            gt_classes: List of ground truth class indices [M]
        """
        for boxes, scores, classes, gt_boxes_img, gt_classes_img in zip(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes
        ):
            self._update_single_image(boxes, scores, classes, gt_boxes_img, gt_classes_img)
    
    def _update_single_image(self, 
                            pred_boxes: torch.Tensor,
                            pred_scores: torch.Tensor,
                            pred_classes: torch.Tensor,
                            gt_boxes: torch.Tensor,
                            gt_classes: torch.Tensor):
        """Update metrics for a single image"""
        if len(pred_boxes) == 0:
            # All predictions are false negatives
            for class_idx in gt_classes.unique():
                self.fn[int(class_idx)].extend([1] * (gt_classes == class_idx).sum().item())
                self.gt_counts[int(class_idx)] += (gt_classes == class_idx).sum().item()
            return
        
        if len(gt_boxes) == 0:
            # All predictions are false positives
            for box, score, class_idx in zip(pred_boxes, pred_scores, pred_classes):
                self.fp[int(class_idx)].append(1)
                self.scores[int(class_idx)].append(score.item())
            return
        
        # Sort predictions by confidence score (descending)
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        pred_classes = pred_classes[sorted_indices]
        
        # Track which ground truths have been matched
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        # For each prediction
        for box, score, class_idx in zip(pred_boxes, pred_scores, pred_classes):
            class_idx = int(class_idx)
            
            # Find ground truths of the same class
            same_class_mask = (gt_classes == class_idx)
            if not same_class_mask.any():
                # No ground truth of this class: false positive
                self.fp[class_idx].append(1)
                self.scores[class_idx].append(score.item())
                continue
            
            # Calculate IoU with all ground truths of the same class
            gt_boxes_class = gt_boxes[same_class_mask]
            ious = self._box_iou(box.unsqueeze(0), gt_boxes_class).squeeze(0)
            
            # Get the best matching ground truth
            best_iou, best_idx = ious.max(dim=0)
            best_idx_global = torch.where(same_class_mask)[0][best_idx]
            
            if best_iou >= self.iou_threshold and not gt_matched[best_idx_global]:
                # True positive: prediction matches an unmatched ground truth
                self.tp[class_idx].append(1)
                self.fp[class_idx].append(0)
                gt_matched[best_idx_global] = True
            else:
                # False positive: no match or ground truth already matched
                self.tp[class_idx].append(0)
                self.fp[class_idx].append(1)
            
            self.scores[class_idx].append(score.item())
        
        # False negatives: unmatched ground truths
        for i, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if not gt_matched[i]:
                class_idx = int(gt_class)
                self.fn[class_idx].append(1)
                self.gt_counts[class_idx] += 1
            else:
                class_idx = int(gt_class)
                self.gt_counts[class_idx] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        
        # Per-class metrics
        for class_idx in range(self.num_classes):
            tp = np.array(self.tp.get(class_idx, []))
            fp = np.array(self.fp.get(class_idx, []))
            fn = np.array(self.fn.get(class_idx, []))
            scores = np.array(self.scores.get(class_idx, []))
            
            if len(scores) == 0:
                continue
            
            # Sort by confidence score
            sorted_indices = np.argsort(-scores)
            tp = tp[sorted_indices]
            fp = fp[sorted_indices]
            scores = scores[sorted_indices]
            
            # Calculate precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
            recall = tp_cumsum / (self.gt_counts.get(class_idx, 0) + 1e-7)
            
            # Average Precision (AP)
            ap = self._compute_ap(precision, recall)
            metrics[f'AP_{class_idx}'] = ap
        
        # Calculate mAP
        aps = [metrics.get(f'AP_{i}', 0) for i in range(self.num_classes)]
        metrics['mAP'] = np.mean(aps) if aps else 0
        
        # Calculate mAP@50 and mAP@75
        metrics.update(self._compute_map_at_iou())
        
        return metrics
    
    def _compute_map_at_iou(self) -> Dict[str, float]:
        """Compute mAP at different IoU thresholds"""
        # This is a simplified version
        # In practice, you would recompute metrics at different thresholds
        return {
            'mAP@50': self._compute_single_map_at_threshold(0.5),
            'mAP@75': self._compute_single_map_at_threshold(0.75),
            'mAP@95': self._compute_single_map_at_threshold(0.95),
        }
    
    def _compute_single_map_at_threshold(self, iou_threshold: float) -> float:
        """Compute mAP at a specific IoU threshold"""
        original_threshold = self.iou_threshold
        self.iou_threshold = iou_threshold
        
        # Recompute metrics with new threshold
        temp_tp = self.tp.copy()
        temp_fp = self.fp.copy()
        temp_fn = self.fn.copy()
        
        # For simplicity, we'll compute a rough estimate
        # In practice, you should recompute from scratch
        aps = []
        for class_idx in range(self.num_classes):
            if class_idx in self.tp:
                # Simplified AP calculation
                tp = np.array(self.tp[class_idx])
                fp = np.array(self.fp[class_idx])
                scores = np.array(self.scores[class_idx])
                
                if len(scores) > 0:
                    sorted_indices = np.argsort(-scores)
                    tp = tp[sorted_indices]
                    fp = fp[sorted_indices]
                    
                    tp_cumsum = np.cumsum(tp)
                    fp_cumsum = np.cumsum(fp)
                    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
                    recall = tp_cumsum / (self.gt_counts.get(class_idx, 0) + 1e-7)
                    
                    ap = self._compute_ap(precision, recall)
                    aps.append(ap)
        
        self.iou_threshold = original_threshold
        
        return np.mean(aps) if aps else 0
    
    @staticmethod
    def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute Average Precision using 101-point interpolation"""
        # Ensure precision and recall are monotonic
        precision = np.concatenate([[0], precision, [0]])
        recall = np.concatenate([[0], recall, [1]])
        
        # Make precision monotonic decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 101-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 101):
            if np.any(recall >= t):
                ap += np.max(precision[recall >= t])
        
        return ap / 101
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        # box1: [N, 4], box2: [M, 4] in xyxy format
        
        # Calculate intersection
        inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
        inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
        inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
        inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        return iou


class ModelEvaluator:
    """Complete model evaluator"""
    def __init__(self, model, device='cuda', iou_threshold=0.5):
        self.model = model.to(device)
        self.device = device
        self.metrics = DetectionMetrics(
            num_classes=model.head.detect.num_classes,
            iou_threshold=iou_threshold
        )
    
    def evaluate(self, dataloader, conf_threshold=0.25, nms_threshold=0.45):
        """Evaluate model on a dataloader"""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Decode predictions
                detections = self._decode_predictions(
                    predictions, 
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                
                # Prepare ground truth
                gt_boxes, gt_classes = self._prepare_ground_truth(targets)
                
                # Update metrics
                pred_boxes = [det['boxes'] for det in detections]
                pred_scores = [det['scores'] for det in detections]
                pred_classes = [det['classes'] for det in detections]
                
                self.metrics.update(
                    pred_boxes, pred_scores, pred_classes,
                    gt_boxes, gt_classes
                )
                
                if batch_idx % 10 == 0:
                    print(f'Processed {batch_idx}/{len(dataloader)} batches')
        
        # Compute final metrics
        final_metrics = self.metrics.compute()
        return final_metrics
    
    def _decode_predictions(self, predictions, conf_threshold=0.25, nms_threshold=0.45):
        """Decode model predictions to detections"""
        cls_outputs, reg_outputs = predictions
        batch_size = cls_outputs[0].shape[0]
        
        detections = []
        
        for batch_idx in range(batch_size):
            batch_detections = {
                'boxes': [],
                'scores': [],
                'classes': []
            }
            
            # Process each scale
            for scale_idx, (cls_pred, reg_pred) in enumerate(zip(cls_outputs, reg_outputs)):
                stride = 8 * (2 ** scale_idx)
                grid_h, grid_w = cls_pred.shape[2:]
                
                # Get predictions for this batch
                cls_pred_scale = cls_pred[batch_idx]  # [C, H, W]
                reg_pred_scale = reg_pred[batch_idx]  # [4, H, W]
                
                # Create grid
                grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
                grid = torch.stack([grid_x, grid_y], dim=0).float().to(self.device)
                
                # Decode boxes
                boxes = self._decode_boxes_single_scale(
                    reg_pred_scale, grid, stride
                )
                
                # Get class predictions
                cls_scores, cls_indices = torch.max(
                    torch.sigmoid(cls_pred_scale), dim=0
                )
                
                # Objectness score (simplified as average of center predictions)
                obj_score = torch.sigmoid(
                    (reg_pred_scale[0] + reg_pred_scale[1]) / 2
                )
                
                # Combine scores
                scores = cls_scores * obj_score
                
                # Filter by confidence
                mask = scores > conf_threshold
                if not mask.any():
                    continue
                
                boxes = boxes.view(-1, 4)[mask.view(-1)]
                scores = scores[mask]
                classes = cls_indices[mask]
                
                # Apply NMS
                keep = self._nms(boxes, scores, nms_threshold)
                
                batch_detections['boxes'].append(boxes[keep])
                batch_detections['scores'].append(scores[keep])
                batch_detections['classes'].append(classes[keep])
            
            # Concatenate detections from all scales
            if batch_detections['boxes']:
                batch_detections['boxes'] = torch.cat(batch_detections['boxes'], dim=0)
                batch_detections['scores'] = torch.cat(batch_detections['scores'], dim=0)
                batch_detections['classes'] = torch.cat(batch_detections['classes'], dim=0)
            else:
                batch_detections['boxes'] = torch.tensor([], device=self.device)
                batch_detections['scores'] = torch.tensor([], device=self.device)
                batch_detections['classes'] = torch.tensor([], device=self.device, dtype=torch.long)
            
            detections.append(batch_detections)
        
        return detections
    
    def _decode_boxes_single_scale(self, reg_pred, grid, stride):
        """Decode boxes for a single scale"""
        # reg_pred: [4, H, W]
        # grid: [2, H, W]
        
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
    
    @staticmethod
    def _nms(boxes, scores, threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Sort by scores
        sorted_scores, indices = torch.sort(scores, descending=True)
        boxes = boxes[indices]
        
        keep = []
        while len(boxes) > 0:
            # Keep the box with highest score
            keep.append(indices[0].item())
            
            if len(boxes) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = ModelEvaluator._box_iou_single(boxes[0:1], boxes[1:]).squeeze(0)
            
            # Remove boxes with high IoU
            mask = ious < threshold
            boxes = boxes[1:][mask]
            indices = indices[1:][mask]
            sorted_scores = sorted_scores[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    @staticmethod
    def _box_iou_single(box1, box2):
        """Calculate IoU between box1 and box2"""
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def _prepare_ground_truth(self, targets):
        """Prepare ground truth for evaluation"""
        gt_boxes = []
        gt_classes = []
        
        for target in targets:
            # Assuming targets are in format [batch_idx, class_idx, x1, y1, x2, y2]
            if len(target) == 0:
                gt_boxes.append(torch.tensor([], device=self.device))
                gt_classes.append(torch.tensor([], device=self.device, dtype=torch.long))
                continue
            
            boxes = target[:, 2:6].to(self.device)
            classes = target[:, 1].long().to(self.device)
            
            gt_boxes.append(boxes)
            gt_classes.append(classes)
        
        return gt_boxes, gt_classes
