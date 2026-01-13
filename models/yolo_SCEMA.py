import torch
import torch.nn as nn
from .backbone import YOLOSCEMA_Backbone
from .neck import YOLOSCEMA_Neck
from .head import YOLOSCEMA_Head


class YOLO_SCEMA(nn.Module):
    """Complete YOLO-SCEMA Model"""
    def __init__(self, 
                 in_channels=3,
                 num_classes=80,
                 depth_multiple=1.0,
                 width_multiple=1.0):
        super(YOLO_SCEMA, self).__init__()
        
        # Calculate channel dimensions
        base_channels = int(64 * width_multiple)
        channels_list = [
            base_channels * 2,  # /4
            base_channels * 4,  # /8
            base_channels * 8,  # /16
            base_channels * 16  # /32
        ]
        
        # Build model components
        self.backbone = YOLOSCEMA_Backbone(in_channels, depth_multiple, width_multiple)
        self.neck = YOLOSCEMA_Neck(channels_list, depth_multiple, width_multiple)
        self.head = YOLOSCEMA_Head(num_classes, [channels_list[1], channels_list[2], channels_list[3]])
        
        # Detection parameters
        self.grid_sizes = None
        self.anchors = None
        
    def forward(self, x):
        # Extract features
        backbone_features = self.backbone(x)
        
        # FPN/PAN neck
        neck_features = self.neck(backbone_features)
        
        # Detection heads
        cls_outputs, reg_outputs = self.head(neck_features)
        
        return cls_outputs, reg_outputs
    
    def init_anchors(self, img_size=640, device='cuda'):
        """Initialize anchor grids"""
        grid_sizes = []
        for stride in [8, 16, 32]:
            grid_size = img_size // stride
            grid_sizes.append((grid_size, grid_size))
        
        self.grid_sizes = grid_sizes
        self.anchors = self.head.detect.make_anchors(grid_sizes, device)
    
    @staticmethod
    def build_model(model_size='n', num_classes=80):
        """Factory method to build different sizes of YOLO-SCEMA"""
        configs = {
            'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
            's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
            'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},
            'l': {'depth_multiple': 1.0, 'width_multiple': 1.0},
            'x': {'depth_multiple': 1.33, 'width_multiple': 1.25}
        }
        
        if model_size not in configs:
            raise ValueError(f"Model size {model_size} not supported. Choose from {list(configs.keys())}")
        
        config = configs[model_size]
        return YOLO_SCEMA(
            in_channels=3,
            num_classes=num_classes,
            depth_multiple=config['depth_multiple'],
            width_multiple=config['width_multiple']
        )
