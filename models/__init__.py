"""
YOLO-SCEMA Model Package

This package contains the implementation of the YOLO-SCEMA architecture,
including the Spatial-Channel Enhanced Multiscale Attention (SCEMA) module
and its integration into the YOLOv8 framework.

Author: Mohammed MAIZA
Repository: https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention
Paper: "Efficient Multiscale Attention with Spatial–Channel Reconstruction for Lightweight Object Detection"
"""

from .backbone import (
    ConvBnSiLU,
    CBS_FR,
    CBS_CSL,
    Bottleneck_FR,
    C2f_FR,
    SPPF_FR,
    YOLOSCEMA_Backbone,
    build_backbone
)

from .attention import (
    SpatialReconstructionUnit,
    ChannelReconstructionUnit,
    CrossSpatialLearning,
    SCEMA,
    ChannelAttention,
    SpatialAttention
)

from .neck import YOLOSCEMA_Neck, build_neck
from .head import Detect_SCEMA, YOLOSCEMA_Head, build_head
from .yolo_SCEMA import YOLO_SCEMA, build_model, get_model_variant

__version__ = "1.0.0"
__author__ = "Mohammed MAIZA"
__email__ = "maiza.mohammed@univ-oran1.dz"
__repository__ = "https://github.com/MAIZA-MOHAMMED/Spatial-Channel-Enhanced-Multiscale-Attention"

__all__ = [
    # Backbone components
    'ConvBnSiLU',
    'CBS_FR',
    'CBS_CSL',
    'Bottleneck_FR',
    'C2f_FR',
    'SPPF_FR',
    'YOLOSCEMA_Backbone',
    'build_backbone',
    
    # Attention modules
    'SpatialReconstructionUnit',
    'ChannelReconstructionUnit',
    'CrossSpatialLearning',
    'SCEMA',
    'ChannelAttention',
    'SpatialAttention',
    
    # Neck components
    'YOLOSCEMA_Neck',
    'build_neck',
    
    # Head components
    'Detect_SCEMA',
    'YOLOSCEMA_Head',
    'build_head',
    
    # Complete model
    'YOLO_SCEMA',
    'build_model',
    'get_model_variant'
]

# Convenience function for model building
def create_model(
    model_size: str = "n",
    num_classes: int = 80,
    pretrained: bool = False,
    pretrained_path: str = None,
    config: dict = None
):
    """
    Convenience function to create a YOLO-SCEMA model.
    
    Args:
        model_size: Model size variant (n, s, m, l, x)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights
        config: Custom configuration dictionary
    
    Returns:
        YOLO-SCEMA model instance
    """
    model = build_model(model_size, num_classes, config)
    
    if pretrained and pretrained_path:
        import torch
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model
