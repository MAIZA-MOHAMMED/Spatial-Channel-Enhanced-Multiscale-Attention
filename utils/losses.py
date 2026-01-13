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
            target_boxes = match_info['target_boxes'][match_info['target_indices']]
            
            # Calculate CIoU loss
            ciou_loss = self.ciou_loss(pos_pred_boxes, target_boxes)
            box_loss += ciou_loss.sum()
            num_positives += len(pos_pred_boxes)
        
        if num_positives > 0:
            box_loss = box_loss / num_positives
        
        return box_loss
    
    def calculate_cls_loss(self, cls_pred, targets, matched_indices):
        """Calculate classification loss"""
        batch_size = cls_pred.shape[0]
        cls_loss = torch.tensor(0.0, device=cls_pred.device)
        num_positives = 0
        
        for batch_idx in range(batch_size):
            match_info = matched_indices[batch_idx]
            if not match_info['positive_mask'].any():
                continue
            
            # Get matched predictions
            pos_cls_pred = cls_pred[batch_idx][match_info['positive_mask']]
            target_classes = match_info['target_classes'][match_info['target_indices']]
            
            # Create one-hot labels
            target_one_hot = torch.zeros_like(pos_cls_pred)
            target_one_hot.scatter_(1, target_classes.unsqueeze(1), 1.0)
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                target_one_hot = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            
            # Calculate classification loss
            if self.focal_gamma > 0:
                # Focal loss
                ce_loss = F.binary_cross_entropy_with_logits(pos_cls_pred, target_one_hot, reduction='none')
                p_t = torch.exp(-ce_loss)
                focal_weight = (1 - p_t) ** self.focal_gamma
                cls_loss += (focal_weight * ce_loss).sum()
            else:
                # Standard BCE
                cls_loss += self.bce_cls(pos_cls_pred, target_one_hot).sum()
            
            num_positives += len(pos_cls_pred)
        
        if num_positives > 0:
            cls_loss = cls_loss / num_positives
        
        return cls_loss
    
    def calculate_obj_loss(self, cls_pred, reg_pred, targets, matched_indices):
        """Calculate objectness loss"""
        batch_size = cls_pred.shape[0]
        obj_loss = torch.tensor(0.0, device=cls_pred.device)
        
        for batch_idx in range(batch_size):
            match_info = matched_indices[batch_idx]
            
            # Objectness target: 1 for positive anchors, 0 for negative
            obj_target = torch.zeros(cls_pred.shape[1], device=cls_pred.device)
            if match_info['positive_mask'].any():
                obj_target[match_info['positive_mask']] = 1.0
            
            # Objectness score from regression (centerness)
            reg_pred_batch = reg_pred[batch_idx]
            obj_score = torch.sigmoid(reg_pred_batch[:, 0] + reg_pred_batch[:, 1]) / 2
            
            # Calculate objectness loss
            obj_loss += self.bce_obj(obj_score, obj_target).mean()
        
        obj_loss = obj_loss / batch_size if batch_size > 0 else obj_loss
        
        return obj_loss
    
    @staticmethod
    def box_iou(box1, box2):
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
    
    @staticmethod
    def ciou_loss(box1, box2):
        """Calculate Complete IoU loss"""
        # box1, box2: [N, 4] in xyxy format
        
        # Convert to center-width-height format
        b1_cx = (box1[:, 0] + box1[:, 2]) / 2
        b1_cy = (box1[:, 1] + box1[:, 3]) / 2
        b1_w = box1[:, 2] - box1[:, 0]
        b1_h = box1[:, 3] - box1[:, 1]
        
        b2_cx = (box2[:, 0] + box2[:, 2]) / 2
        b2_cy = (box2[:, 1] + box2[:, 3]) / 2
        b2_w = box2[:, 2] - box2[:, 0]
        b2_h = box2[:, 3] - box2[:, 1]
        
        # Calculate IoU
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = b1_w * b1_h + b2_w * b2_h - inter_area
        iou = inter_area / (union_area + 1e-7)
        
        # Center distance
        center_dist = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
        
        # Diagonal distance of enclosing box
        c_w = torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0])
        c_h = torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1])
        c_diag = c_w ** 2 + c_h ** 2 + 1e-7
        
        # Aspect ratio
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(b2_w / (b2_h + 1e-7)) - 
                                            torch.atan(b1_w / (b1_h + 1e-7)), 2)
        
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU loss
        ciou = iou - (center_dist / c_diag + alpha * v)
        loss = 1 - ciou
        
        return loss
