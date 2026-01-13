import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOSCEMALoss(nn.Module):
    """Loss function for YOLO-SCEMA"""
    def __init__(self, num_classes=80, box_gain=7.5, cls_gain=0.5, 
                 obj_gain=1.0, label_smoothing=0.0):
        super(YOLOSCEMALoss, self).__init__()
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.label_smoothing = label_smoothing
        
        # BCE with logits loss
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Focal loss parameters
        self.focal_gamma = 0.0  # disable focal loss by default
        
    def forward(self, predictions, targets):
        """Calculate loss"""
        cls_preds, reg_preds = predictions
        
        device = cls_preds[0].device
        total_loss = torch.tensor(0.0, device=device)
        
        # Process each scale
        for scale_idx, (cls_pred, reg_pred) in enumerate(zip(cls_preds, reg_preds)):
            # Get stride for this scale
            stride = 8 * (2 ** scale_idx)
            
            # Reshape predictions
            batch_size, _, grid_h, grid_w = cls_pred.shape
            cls_pred = cls_pred.view(batch_size, self.num_classes, -1).permute(0, 2, 1)
            reg_pred = reg_pred.view(batch_size, 4, -1).permute(0, 2, 1)
            
            # Generate grid
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
            grid = torch.stack([grid_x, grid_y], dim=2).float().to(device)
            grid = grid.view(-1, 2)
            
            # Decode predictions
            pred_boxes = self.decode_boxes(reg_pred, grid, stride)
            
            # Match predictions with targets
            matched_indices = self.match_predictions(pred_boxes, targets, stride)
            
            # Calculate losses
            box_loss = self.calculate_box_loss(reg_pred, pred_boxes, targets, matched_indices, stride)
            cls_loss = self.calculate_cls_loss(cls_pred, targets, matched_indices)
            obj_loss = self.calculate_obj_loss(cls_pred, reg_pred, targets, matched_indices)
            
            # Weighted sum
            scale_loss = (self.box_gain * box_loss + 
                         self.cls_gain * cls_loss + 
                         self.obj_gain * obj_loss)
            
            total_loss += scale_loss
        
        return total_loss
    
    def decode_boxes(self, reg_pred, grid, stride):
        """Decode regression predictions to boxes"""
        # reg_pred: [batch, num_anchors, 4] (dx, dy, dw, dh)
        boxes = torch.zeros_like(reg_pred)
        
        # Center coordinates
        boxes[..., 0:2] = (reg_pred[..., 0:2].sigmoid() + grid) * stride
        
        # Width and height
        boxes[..., 2:4] = torch.exp(reg_pred[..., 2:4]) * stride
        
        # Convert to xyxy format
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
        
        return boxes_xyxy
    
    def match_predictions(self, pred_boxes, targets, stride):
        """Match predictions with ground truth targets"""
        batch_size = pred_boxes.shape[0]
        num_anchors = pred_boxes.shape[1]
        
        matched_indices = []
        
        for batch_idx in range(batch_size):
            batch_targets = targets[targets[:, 0] == batch_idx]
            if len(batch_targets) == 0:
                matched_indices.append(torch.zeros(num_anchors, dtype=torch.bool, device=pred_boxes.device))
                continue
            
            # Calculate IoU between predictions and targets
            target_boxes = batch_targets[:, 2:6]  # xyxy format
            ious = self.box_iou(pred_boxes[batch_idx], target_boxes)
            
            # Find best matches
            max_iou, best_match_idx = ious.max(dim=1)
            
            # Positive anchors: IoU > 0.5
            positive_mask = max_iou > 0.5
            
            matched_indices.append({
                'positive_mask': positive_mask,
                'target_indices': best_match_idx[positive_mask],
                'target_boxes': target_boxes,
                'target_classes': batch_targets[:, 1].long()
            })
        
        return matched_indices
    
    def calculate_box_loss(self, reg_pred, pred_boxes, targets, matched_indices, stride):
        """Calculate bounding box regression loss (CIoU)"""
        batch_size = reg_pred.shape[0]
        box_loss = torch.tensor(0.0, device=reg_pred.device)
        num_positives = 0
        
        for batch_idx in range(batch_size):
            match_info = matched_indices[batch_idx]
            if not match_info['positive_mask'].any():
                continue
            
            # Get matched predictions and targets
            pos_pred_boxes = pred_boxes[batch_idx][match_info['positive_mask']]
            target_boxes = match
