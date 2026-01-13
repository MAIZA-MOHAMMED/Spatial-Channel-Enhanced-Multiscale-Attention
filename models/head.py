import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBnSiLU


class Detect_SCEMA(nn.Module):
    """YOLO-SCEMA Detection Head (Anchor-free)"""
    def __init__(self, num_classes=80, channels_list=[256, 512, 1024], stride=[8, 16, 32]):
        super(Detect_SCEMA, self).__init__()
        self.num_classes = num_classes
        self.stride = stride
        
        # Number of outputs per anchor
        self.num_outputs = num_classes + 4  # cls + bbox
        
        # Detection heads for different scales
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        for in_channels in channels_list:
            # Classification head
            cls_head = nn.Sequential(
                ConvBnSiLU(in_channels, in_channels, 3, 1),
                ConvBnSiLU(in_channels, in_channels, 3, 1),
                nn.Conv2d(in_channels, self.num_classes, 1, 1, 0)
            )
            
            # Regression head (bbox + objectness)
            reg_head = nn.Sequential(
                ConvBnSiLU(in_channels, in_channels, 3, 1),
                ConvBnSiLU(in_channels, in_channels, 3, 1),
                nn.Conv2d(in_channels, 4, 1, 1, 0)
            )
            
            self.cls_preds.append(cls_head)
            self.reg_preds.append(reg_head)
        
        # Initialize biases
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights and biases"""
        for cls_head, reg_head in zip(self.cls_preds, self.reg_preds):
            # Classification head
            pi = 0.01  # probability initialization
            b = -torch.log(torch.tensor((1 - pi) / pi))
            nn.init.constant_(cls_head[-1].bias, b)
            
            # Regression head
            nn.init.constant_(reg_head[-1].bias, 0)
    
    def forward(self, features):
        """Forward pass"""
        cls_outputs = []
        reg_outputs = []
        
        for i, x in enumerate(features):
            cls_output = self.cls_preds[i](x)
            reg_output = self.reg_preds[i](x)
            
            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)
        
        return cls_outputs, reg_outputs
    
    def make_anchors(self, grid_sizes, device):
        """Generate anchor points for anchor-free detection"""
        anchors = []
        for i, (grid_h, grid_w) in enumerate(grid_sizes):
            # Create grid
            y, x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
            grid = torch.stack([x, y], dim=-1).float().to(device)
            
            # Normalize grid coordinates
            grid = grid.view(-1, 2)
            anchors.append(grid)
        
        return anchors


class YOLOSCEMA_Head(nn.Module):
    """Complete YOLO-SCEMA Head"""
    def __init__(self, num_classes=80, channels_list=[256, 512, 1024]):
        super(YOLOSCEMA_Head, self).__init__()
        self.detect = Detect_SCEMA(num_classes, channels_list)
        
    def forward(self, features):
        return self.detect(features)
