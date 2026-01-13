import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialReconstructionUnit(nn.Module):
    """SRU: Spatial Reconstruction Unit"""
    def __init__(self, channels, reduction_ratio=16, threshold=0.5):
        super(SpatialReconstructionUnit, self).__init__()
        self.threshold = threshold
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels)
        
        # Spatial attention
        self.conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.conv2 = nn.Conv2d(channels // reduction_ratio, channels, 1)
        
    def forward(self, x):
        # Group normalization
        x_norm = self.gn(x)
        
        # Generate attention weights using sigmoid gate
        avg_out = torch.mean(x_norm, dim=1, keepdim=True)
        max_out, _ = torch.max(x_norm, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv2(F.relu(self.conv1(avg_out + max_out))))
        
        # Gate operation with threshold
        W = (attention > self.threshold).float()
        W1 = W
        W2 = 1 - W
        
        # Weighted features
        X1_W = W1 * x
        X2_W = W2 * x
        
        # Cross-reconstruction
        # Split channels for cross-reconstruction
        c = x.size(1)
        X11_W = X1_W[:, :c//2, :, :]
        X12_W = X1_W[:, c//2:, :, :]
        X21_W = X2_W[:, :c//2, :, :]
        X22_W = X2_W[:, c//2:, :, :]
        
        # Cross reconstruction
        X_W1 = X11_W + X22_W
        X_W2 = X21_W + X12_W
        
        # Concatenation
        X_W = torch.cat([X_W1, X_W2], dim=1)
        
        return X_W


class ChannelReconstructionUnit(nn.Module):
    """CRU: Channel Reconstruction Unit"""
    def __init__(self, channels, alpha=0.5, compression_ratio=2, groups=2, kernel_size=3):
        super(ChannelReconstructionUnit, self).__init__()
        self.alpha = alpha
        self.r = compression_ratio
        self.g = groups
        self.k = kernel_size
        
        # Split channels
        self.upper_channels = int(alpha * channels)
        self.lower_channels = channels - self.upper_channels
        
        # Upper branch (rich feature extractor)
        self.upper_conv1 = nn.Conv2d(self.upper_channels // self.r, 
                                     self.upper_channels // self.r, 
                                     kernel_size=kernel_size, 
                                     padding=kernel_size//2, 
                                     groups=self.g)
        self.upper_conv2 = nn.Conv2d(self.upper_channels // self.r, 
                                     channels, 
                                     kernel_size=1)
        
        # Lower branch (efficient processing)
        self.lower_conv = nn.Conv2d(self.lower_channels // self.r, 
                                    channels - (self.lower_channels // self.r), 
                                    kernel_size=1)
        
        # Channel attention for fusion
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels * 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Split channels
        X_up = x[:, :self.upper_channels, :, :]
        X_low = x[:, self.upper_channels:, :, :]
        
        # Channel compression
        X_up_compressed = F.adaptive_avg_pool2d(X_up, (height // self.r, width // self.r))
        X_low_compressed = F.adaptive_avg_pool2d(X_low, (height // self.r, width // self.r))
        
        # Upper branch transformation
        Y1 = self.upper_conv1(X_up_compressed)
        Y1 = self.upper_conv2(Y1)
        Y1 = F.interpolate(Y1, size=(height, width), mode='bilinear', align_corners=False)
        
        # Lower branch transformation
        Y2 = self.lower_conv(X_low_compressed)
        Y2 = torch.cat([Y2, X_low_compressed], dim=1)
        Y2 = F.interpolate(Y2, size=(height, width), mode='bilinear', align_corners=False)
        
        # Fusion with channel attention
        S1 = self.gap(Y1).view(batch_size, -1)
        S2 = self.gap(Y2).view(batch_size, -1)
        S = torch.cat([S1, S2], dim=1)
        
        attention = self.fc(S)
        attention = attention.view(batch_size, 2, channels, 1, 1)
        beta1 = attention[:, 0, :, :, :]
        beta2 = attention[:, 1, :, :, :]
        
        Y = beta1 * Y1 + beta2 * Y2
        
        return Y


class CrossSpatialLearning(nn.Module):
    """CSL: Cross Spatial Learning Module"""
    def __init__(self, channels, groups=4):
        super(CrossSpatialLearning, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups
        
        # 1x1 convolution branches (2 branches)
        self.conv1x1_1 = nn.Conv2d(channels, channels, kernel_size=1, groups=groups)
        self.conv1x1_2 = nn.Conv2d(channels, channels, kernel_size=1, groups=groups)
        
        # 3x3 convolution branch
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        
        # Global average pooling for spatial encoding
        self.gap_h = nn.AdaptiveAvgPool2d((None, 1))
        self.gap_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Final convolution
        self.final_conv = nn.Conv2d(channels * 3, channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Feature grouping
        x_groups = torch.chunk(x, self.groups, dim=1)
        
        outputs = []
        for x_group in x_groups:
            # 1x1 branch 1
            x_1x1_1 = self.conv1x1_1(x_group)
            
            # 1x1 branch 2 with coordinate attention
            x_h = self.gap_h(x_group)
            x_w = self.gap_w(x_group)
            x_cat = torch.cat([x_h, x_w], dim=2)
            x_cat = x_cat.view(batch_size, self.channels_per_group, 2, -1)
            x_1x1_2 = self.conv1x1_2(x_cat)
            x_1x1_2 = x_1x1_2.view(batch_size, self.channels_per_group, height, width)
            
            # 3x3 branch
            x_3x3 = self.conv3x3(x_group)
            
            # Cross spatial learning with matrix dot product
            attention_map = torch.sigmoid(torch.matmul(
                x_1x1_1.view(batch_size, self.channels_per_group, -1).transpose(1, 2),
                x_3x3.view(batch_size, self.channels_per_group, -1)
            ))
            
            # Apply attention
            x_attended = torch.matmul(
                attention_map,
                x_1x1_2.view(batch_size, self.channels_per_group, -1)
            ).view(batch_size, self.channels_per_group, height, width)
            
            # Concatenate all features
            x_group_out = torch.cat([x_1x1_1, x_1x1_2, x_attended], dim=1)
            outputs.append(x_group_out)
        
        # Merge groups
        x_out = torch.cat(outputs, dim=1)
        x_out = self.final_conv(x_out)
        
        return x_out


class SCEMA(nn.Module):
    """Spatial-Channel Enhanced Multiscale Attention"""
    def __init__(self, channels, reduction_ratio=16, alpha=0.5, 
                 compression_ratio=2, groups=4, kernel_size=3):
        super(SCEMA, self).__init__()
        
        # Left branch: Feature Refinement Module
        self.sru = SpatialReconstructionUnit(channels, reduction_ratio)
        self.cru = ChannelReconstructionUnit(channels, alpha, compression_ratio, groups, kernel_size)
        
        # Right branch: Cross Spatial Learning Module
        self.csl = CrossSpatialLearning(channels, groups)
        
        # CBAM attention (optional fusion)
        self.cbam_channel = ChannelAttention(channels, reduction_ratio)
        self.cbam_spatial = SpatialAttention(kernel_size=7)
        
        # Final fusion
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        
    def forward(self, x):
        # Left branch
        x_left = self.sru(x)
        x_left = self.cru(x_left)
        x_left = self.cbam_channel(x_left)
        x_left = self.cbam_spatial(x_left)
        
        # Right branch
        x_right = self.csl(x)
        
        # Fusion
        x_out = torch.cat([x_left, x_right], dim=1)
        x_out = self.fusion_conv(x_out)
        
        return x_out


# CBAM components (as referenced in SCEMA)
class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        return x * self.sigmoid(attention)
