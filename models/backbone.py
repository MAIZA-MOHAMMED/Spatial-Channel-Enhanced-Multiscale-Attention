"""
YOLO-SCEMA Backbone Implementation

This module implements the backbone network for YOLO-SCEMA,
integrating the SCEMA attention modules into the YOLOv8 architecture.

Key Components:
1. ConvBnSiLU: Basic convolutional block
2. CBS_FR: Conv block with Feature Refinement (SCEMA)
3. CBS_CSL: Conv block with Cross Spatial Learning
4. Bottleneck_FR: Bottleneck block with Feature Refinement
5. C2f_FR: C2f block with Feature Refinement
6. SPPF_FR: SPPF block with Feature Refinement
7. YOLOSCEMA_Backbone: Complete backbone network

Author: Mohammed MAIZA
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from .attention import SCEMA


class ConvBnSiLU(nn.Module):
    """
    Basic Convolution-BatchNorm-SiLU block (CBS)
    
    This is the fundamental building block used throughout the YOLO-SCEMA architecture.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        groups: Number of groups for grouped convolution
        bias: Whether to use bias in convolution
    
    Reference:
        YOLOv8 architecture from Ultralytics
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super(ConvBnSiLU, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block"""
        return self.act(self.bn(self.conv(x)))
    
    def get_flops(self, input_size: Tuple[int, int]) -> int:
        """
        Calculate FLOPs for this block
        
        Args:
            input_size: Input feature map size (height, width)
        
        Returns:
            Number of floating point operations
        """
        h, w = input_size
        kernel_h = kernel_w = self.conv.kernel_size[0]
        stride_h = stride_w = self.conv.stride[0]
        
        # Convolution FLOPs
        output_h = (h + 2 * self.conv.padding[0] - kernel_h) // stride_h + 1
        output_w = (w + 2 * self.conv.padding[1] - kernel_w) // stride_w + 1
        
        conv_flops = (
            self.conv.in_channels * self.conv.out_channels *
            kernel_h * kernel_w * output_h * output_w
        )
        
        # BatchNorm FLOPs (approximate)
        bn_flops = 4 * self.conv.out_channels * output_h * output_w
        
        return conv_flops + bn_flops


class CBS_FR(nn.Module):
    """
    Convolutional Block with Feature Refinement (SCEMA enhanced)
    
    This block integrates the SCEMA module after the basic convolution
    to perform spatial and channel feature refinement.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        scema_config: Configuration dictionary for SCEMA module
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        scema_config: Dict[str, Any] = None
    ):
        super(CBS_FR, self).__init__()
        
        self.conv = ConvBnSiLU(in_channels, out_channels, kernel_size, stride)
        
        # Default SCEMA configuration
        default_scema_config = {
            'reduction_ratio': 16,
            'alpha': 0.5,
            'compression_ratio': 2,
            'groups': 4,
            'kernel_size': 3,
            'threshold': 0.5
        }
        
        if scema_config:
            default_scema_config.update(scema_config)
        
        # Create SCEMA module for feature refinement
        self.scema = SCEMA(
            channels=out_channels,
            reduction_ratio=default_scema_config['reduction_ratio'],
            alpha=default_scema_config['alpha'],
            compression_ratio=default_scema_config['compression_ratio'],
            groups=default_scema_config['groups'],
            kernel_size=default_scema_config['kernel_size']
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Conv → SCEMA feature refinement
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Refined feature tensor
        """
        # Apply basic convolution
        x = self.conv(x)
        
        # Apply SCEMA feature refinement
        x = self.scema(x)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from the SCEMA module
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of attention maps from different SCEMA components
        """
        # Apply convolution
        x_conv = self.conv(x)
        
        # Hook to capture attention maps
        attention_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                attention_maps[name] = output.detach()
            return hook
        
        # Register hooks on SCEMA submodules
        hooks = []
        for name, module in self.scema.named_modules():
            if isinstance(module, (SpatialReconstructionUnit, 
                                  ChannelReconstructionUnit,
                                  CrossSpatialLearning)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward through SCEMA
        _ = self.scema(x_conv)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps


class CBS_CSL(nn.Module):
    """
    Convolutional Block with Cross Spatial Learning
    
    This block uses only the CSL (Cross Spatial Learning) component
    for efficient multi-scale feature extraction.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1
    ):
        super(CBS_CSL, self).__init__()
        
        self.conv = ConvBnSiLU(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block"""
        return self.conv(x)


class Bottleneck_FR(nn.Module):
    """
    Bottleneck Block with Feature Refinement
    
    Enhanced bottleneck block that integrates SCEMA for feature refinement.
    This block uses a residual connection when input and output channels match.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        shortcut: Whether to use shortcut connection
        groups: Number of groups for grouped convolution
        expansion: Expansion ratio for hidden channels
        scema_config: Configuration for SCEMA module
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        scema_config: Dict[str, Any] = None
    ):
        super(Bottleneck_FR, self).__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.shortcut = shortcut and (in_channels == out_channels)
        
        # First convolution (channel reduction)
        self.cv1 = ConvBnSiLU(in_channels, hidden_channels, 1, 1)
        
        # Second convolution with feature refinement
        self.cv2 = CBS_FR(hidden_channels, out_channels, 3, 1, scema_config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck block
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor with optional residual connection
        """
        # Main path
        out = self.cv2(self.cv1(x))
        
        # Residual connection (if applicable)
        if self.shortcut:
            out = out + x
        
        return out
    
    def get_complexity(self, input_size: Tuple[int, int]) -> Dict[str, int]:
        """
        Calculate computational complexity metrics
        
        Args:
            input_size: Input feature map size (height, width)
        
        Returns:
            Dictionary with parameters and FLOPs
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate FLOPs for each component
        h, w = input_size
        
        # cv1 FLOPs
        cv1_flops = self.cv1.get_flops((h, w))
        
        # cv2 FLOPs (after cv1 reduces resolution if stride > 1)
        cv2_input_h = h // self.cv1.conv.stride[0]
        cv2_input_w = w // self.cv1.conv.stride[1]
        cv2_flops = self.cv2.conv.get_flops((cv2_input_h, cv2_input_w))
        
        # SCEMA module FLOPs (approximate)
        scema_flops = self._estimate_scema_flops(out_channels, (cv2_input_h, cv2_input_w))
        
        total_flops = cv1_flops + cv2_flops + scema_flops
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'cv1_flops': cv1_flops,
            'cv2_flops': cv2_flops,
            'scema_flops': scema_flops
        }
    
    def _estimate_scema_flops(self, channels: int, input_size: Tuple[int, int]) -> int:
        """Estimate FLOPs for SCEMA module"""
        h, w = input_size
        
        # Approximate FLOPs for SCEMA operations
        # This is a simplified estimation
        sr_flops = channels * h * w * 10  # Spatial reconstruction
        cr_flops = channels * h * w * 20  # Channel reconstruction
        csl_flops = channels * h * w * 15  # Cross spatial learning
        
        return sr_flops + cr_flops + csl_flops


class C2f_FR(nn.Module):
    """
    C2f Block with Feature Refinement
    
    Enhanced C2f (Cross Stage Partial network with 2 convolutions) block
    that integrates multiple bottleneck blocks with feature refinement.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n: Number of bottleneck blocks
        shortcut: Whether to use shortcut connections in bottlenecks
        expansion: Expansion ratio for bottleneck hidden channels
        scema_config: Configuration for SCEMA modules
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = False,
        expansion: float = 0.5,
        scema_config: Dict[str, Any] = None
    ):
        super(C2f_FR, self).__init__()
        
        self.c = out_channels // 2  # Hidden channel size
        
        # First convolution splits channels
        self.cv1 = ConvBnSiLU(in_channels, 2 * self.c, 1, 1)
        
        # Multiple bottleneck blocks
        self.m = nn.ModuleList([
            Bottleneck_FR(
                self.c,
                self.c,
                shortcut=shortcut,
                expansion=expansion,
                scema_config=scema_config
            )
            for _ in range(n)
        ])
        
        # Final convolution
        self.cv2 = ConvBnSiLU((2 + n) * self.c, out_channels, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f block
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor with enriched features
        """
        # Split channels after first convolution
        y = list(self.cv1(x).chunk(2, dim=1))
        
        # Process through bottleneck blocks
        for m in self.m:
            y.append(m(y[-1]))
        
        # Concatenate and final convolution
        return self.cv2(torch.cat(y, dim=1))
    
    def get_feature_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features at different stages of the block
        
        Args:
            x: Input tensor
        
        Returns:
            List of feature tensors at different processing stages
        """
        features = []
        
        # Initial features
        initial = self.cv1(x)
        features.append(initial)
        
        # Split and process
        y = list(initial.chunk(2, dim=1))
        
        # Collect features after each bottleneck
        for i, m in enumerate(self.m):
            y.append(m(y[-1]))
            features.append(torch.cat(y, dim=1))
        
        # Final features
        final = self.cv2(features[-1])
        features.append(final)
        
        return features


class SPPF_FR(nn.Module):
    """
    Spatial Pyramid Pooling Fast with Feature Refinement
    
    Enhanced SPPF block that integrates SCEMA modules for improved
    multi-scale feature extraction.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Max pooling kernel size
        scema_config: Configuration for SCEMA modules
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        scema_config: Dict[str, Any] = None
    ):
        super(SPPF_FR, self).__init__()
        
        # Intermediate channels
        mid_channels = in_channels // 2
        
        # First convolution with feature refinement
        self.cv1 = CBS_FR(in_channels, mid_channels, 1, 1, scema_config)
        
        # Max pooling layers with different kernel sizes
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # Final convolution with feature refinement
        self.cv2 = CBS_FR(mid_channels * 4, out_channels, 1, 1, scema_config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SPPF block
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Multi-scale features with spatial pyramid pooling
        """
        # Initial feature refinement
        x = self.cv1(x)
        
        # Spatial pyramid pooling
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([x, y1, y2, y3], dim=1)
        
        # Final feature refinement
        return self.cv2(multi_scale_features)
    
    def get_pooling_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from each pooling level
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of features from different pooling scales
        """
        # Initial features
        x_refined = self.cv1(x)
        
        # Pooling features
        pool1 = self.maxpool(x_refined)
        pool2 = self.maxpool(pool1)
        pool3 = self.maxpool(pool2)
        
        # Final features
        concatenated = torch.cat([x_refined, pool1, pool2, pool3], dim=1)
        final = self.cv2(concatenated)
        
        return {
            'initial': x_refined,
            'pool1': pool1,
            'pool2': pool2,
            'pool3': pool3,
            'concatenated': concatenated,
            'final': final
        }


class YOLOSCEMA_Backbone(nn.Module):
    """
    YOLO-SCEMA Backbone Network
    
    Complete backbone architecture integrating SCEMA modules into YOLOv8.
    This backbone extracts multi-scale features at different resolutions.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        depth_multiple: Depth scaling factor
        width_multiple: Width (channel) scaling factor
        scema_config: Configuration for SCEMA modules
        variant: Model variant (n, s, m, l, x)
    
    Output Features:
        - feat1: /4 resolution features
        - feat2: /8 resolution features  
        - feat3: /16 resolution features
        - feat4: /32 resolution features
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        scema_config: Dict[str, Any] = None,
        variant: str = "n"
    ):
        super(YOLOSCEMA_Backbone, self).__init__()
        
        self.variant = variant
        self.in_channels = in_channels
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        
        # Base channel calculations
        base_channels = int(64 * width_multiple)
        
        # Store feature channel dimensions
        self.channels = [
            base_channels * 2,   # /4
            base_channels * 4,   # /8
            base_channels * 8,   # /16
            base_channels * 16   # /32
        ]
        
        # -------------------- Stage 0: Stem --------------------
        self.stem = ConvBnSiLU(in_channels, base_channels, 3, 2)
        
        # -------------------- Stage 1: /4 features --------------------
        self.stage1 = nn.Sequential(
            CBS_FR(base_channels, base_channels * 2, 3, 2, scema_config),
            C2f_FR(
                base_channels * 2,
                base_channels * 2,
                n=max(round(3 * depth_multiple), 1),
                shortcut=True,
                scema_config=scema_config
            )
        )
        
        # -------------------- Stage 2: /8 features --------------------
        self.stage2 = nn.Sequential(
            CBS_FR(base_channels * 2, base_channels * 4, 3, 2, scema_config),
            C2f_FR(
                base_channels * 4,
                base_channels * 4,
                n=max(round(6 * depth_multiple), 1),
                shortcut=True,
                scema_config=scema_config
            )
        )
        
        # -------------------- Stage 3: /16 features --------------------
        self.stage3 = nn.Sequential(
            CBS_FR(base_channels * 4, base_channels * 8, 3, 2, scema_config),
            C2f_FR(
                base_channels * 8,
                base_channels * 8,
                n=max(round(6 * depth_multiple), 1),
                shortcut=True,
                scema_config=scema_config
            )
        )
        
        # -------------------- Stage 4: /32 features --------------------
        self.stage4 = nn.Sequential(
            CBS_FR(base_channels * 8, base_channels * 16, 3, 2, scema_config),
            C2f_FR(
                base_channels * 16,
                base_channels * 16,
                n=max(round(3 * depth_multiple), 1),
                shortcut=True,
                scema_config=scema_config
            ),
            SPPF_FR(base_channels * 16, base_channels * 16, scema_config=scema_config)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the backbone
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            List of feature tensors at different scales:
            [feat1, feat2, feat3, feat4] corresponding to strides [4, 8, 16, 32]
        """
        # Stem: /2 resolution
        x0 = self.stem(x)          # [B, base_channels, H/2, W/2]
        
        # Stage 1: /4 resolution
        x1 = self.stage1(x0)       # [B, base_channels*2, H/4, W/4]
        
        # Stage 2: /8 resolution
        x2 = self.stage2(x1)       # [B, base_channels*4, H/8, W/8]
        
        # Stage 3: /16 resolution
        x3 = self.stage3(x2)       # [B, base_channels*8, H/16, W/16]
        
        # Stage 4: /32 resolution
        x4 = self.stage4(x3)       # [B, base_channels*16, H/32, W/32]
        
        return [x1, x2, x3, x4]
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from all intermediate layers
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of features from all layers
        """
        features = {}
        
        # Stem features
        features['stem'] = self.stem(x)
        
        # Stage 1 features
        x1 = self.stage1[0](features['stem'])
        features['stage1_conv'] = x1
        features['stage1_c2f'] = self.stage1[1](x1)
        
        # Stage 2 features
        x2 = self.stage2[0](features['stage1_c2f'])
        features['stage2_conv'] = x2
        features['stage2_c2f'] = self.stage2[1](x2)
        
        # Stage 3 features
        x3 = self.stage3[0](features['stage2_c2f'])
        features['stage3_conv'] = x3
        features['stage3_c2f'] = self.stage3[1](x3)
        
        # Stage 4 features
        x4 = self.stage4[0](features['stage3_c2f'])
        features['stage4_conv'] = x4
        x4_c2f = self.stage4[1](x4)
        features['stage4_c2f'] = x4_c2f
        features['stage4_sppf'] = self.stage4[2](x4_c2f)
        
        return features
    
    def get_attention_statistics(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Collect attention statistics from all SCEMA modules
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with attention statistics
        """
        stats = {}
        
        # Forward pass to collect features
        features = self.get_feature_maps(x)
        
        # Collect attention from CBS_FR blocks
        attention_maps = {}
        
        def collect_attention(name, module, input, output):
            if hasattr(module, 'get_attention_maps'):
                maps = module.get_attention_maps(input[0])
                attention_maps[name] = maps
        
        # Register hooks on all CBS_FR modules
        hooks = []
        for name, module in self.named_modules():
            if isinstance(module, CBS_FR):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: collect_attention(n, m, i, o)
                )
                hooks.append(hook)
        
        # Trigger forward pass
        _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process attention maps
        for name, maps in attention_maps.items():
            for map_name, map_tensor in maps.items():
                key = f"{name}.{map_name}"
                stats[f"{key}.mean"] = map_tensor.mean().item()
                stats[f"{key}.std"] = map_tensor.std().item()
                stats[f"{key}.min"] = map_tensor.min().item()
                stats[f"{key}.max"] = map_tensor.max().item()
        
        return stats
    
    def get_complexity(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """
        Calculate computational complexity of the backbone
        
        Args:
            input_size: Input image size (height, width)
        
        Returns:
            Dictionary with parameters, FLOPs, and memory usage
        """
        h, w = input_size
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate FLOPs (simplified estimation)
        flops_breakdown = {}
        
        # Stem FLOPs
        flops_breakdown['stem'] = self._estimate_module_flops(self.stem, (h, w))
        
        # Stage FLOPs
        current_h, current_w = h // 2, w // 2  # After stem
        
        for i, stage in enumerate([self.stage1, self.stage2, self.stage3, self.stage4]):
            stage_name = f"stage{i+1}"
            flops_breakdown[stage_name] = self._estimate_module_flops(
                stage, (current_h, current_w)
            )
            current_h //= 2
            current_w //= 2
        
        total_flops = sum(flops_breakdown.values())
        
        # Memory usage estimation
        memory_bytes = total_params * 4  # Assuming float32
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'flops_breakdown': flops_breakdown,
            'memory_bytes': memory_bytes,
            'memory_mb': memory_bytes / (1024 ** 2),
            'variant': self.variant,
            'channels': self.channels
        }
    
    def _estimate_module_flops(self, module: nn.Module, input_size: Tuple[int, int]) -> int:
        """Estimate FLOPs for a module"""
        # Simplified FLOPs estimation
        # In practice, use tools like thop or fvcore
        h, w = input_size
        
        # Count convolutional operations
        conv_flops = 0
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                out_h = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
                out_w = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
                
                conv_flops += (
                    m.in_channels * m.out_channels *
                    m.kernel_size[0] * m.kernel_size[1] *
                    out_h * out_w
                )
        
        # Approximate other operations
        other_flops = h * w * 1000  # Rough estimate
        
        return conv_flops + other_flops
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def build_backbone(
    variant: str = "n",
    in_channels: int = 3,
    scema_config: Dict[str, Any] = None,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None
) -> YOLOSCEMA_Backbone:
    """
    Factory function to build YOLO-SCEMA backbone
    
    Args:
        variant: Model variant (n, s, m, l, x)
        in_channels: Number of input channels
        scema_config: SCEMA module configuration
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights
    
    Returns:
        YOLO-SCEMA backbone instance
    """
    # Model variant configurations
    variant_configs = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},
        'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},
        'l': {'depth_multiple': 1.00, 'width_multiple': 1.00},
        'x': {'depth_multiple': 1.33, 'width_multiple': 1.25}
    }
    
    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_configs.keys())}")
    
    config = variant_configs[variant]
    
    # Build backbone
    backbone = YOLOSCEMA_Backbone(
        in_channels=in_channels,
        depth_multiple=config['depth_multiple'],
        width_multiple=config['width_multiple'],
        scema_config=scema_config,
        variant=variant
    )
    
    # Load pretrained weights if specified
    if pretrained and pretrained_path:
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'backbone' in checkpoint:
                backbone.load_state_dict(checkpoint['backbone'])
            elif 'model_state_dict' in checkpoint:
                # Filter backbone parameters
                backbone_state_dict = {
                    k.replace('backbone.', ''): v
                    for k, v in checkpoint['model_state_dict'].items()
                    if k.startswith('backbone.')
                }
                backbone.load_state_dict(backbone_state_dict)
            else:
                # Assume it's a backbone-only checkpoint
                backbone.load_state_dict(checkpoint)
            
            print(f"Loaded pretrained backbone weights from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    
    return backbone


# Example usage and testing
if __name__ == "__main__":
    # Test the backbone
    print("Testing YOLO-SCEMA Backbone...")
    
    # Create a nano variant backbone
    backbone = build_backbone('n', in_channels=3)
    
    # Create dummy input
    batch_size = 2
    height, width = 640, 640
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    features = backbone(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Number of output features: {len(features)}")
    
    for i, feat in enumerate(features):
        stride = 4 * (2 ** i)
        print(f"Feature {i+1} (stride {stride}): {feat.shape}")
    
    # Get complexity analysis
    complexity = backbone.get_complexity((height, width))
    print(f"\nBackbone Complexity:")
    print(f"  Parameters: {complexity['parameters']:,}")
    print(f"  FLOPs: {complexity['flops']:,}")
    print(f"  Memory: {complexity['memory_mb']:.2f} MB")
    
    # Test attention statistics
    print("\nCollecting attention statistics...")
    stats = backbone.get_attention_statistics(dummy_input)
    print(f"Collected {len(stats)} attention statistics")
    
    print("\nBackbone test completed successfully!")
