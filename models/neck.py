import torch
import torch.nn as nn
from .backbone import ConvBnSiLU, C2f_FR


class YOLOSCEMA_Neck(nn.Module):
    """YOLO-SCEMA Neck with PAN structure"""
    def __init__(self, channels_list, depth_multiple=1.0, width_multiple=1.0):
        super(YOLOSCEMA_Neck, self).__init__()
        
        # Upsample layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down path (FPN-like)
        self.conv1 = ConvBnSiLU(channels_list[3], channels_list[2], 1, 1)
        self.c2f1 = C2f_FR(channels_list[2] * 2, channels_list[2], 
                          n=max(round(3 * depth_multiple), 1), shortcut=False)
        
        self.conv2 = ConvBnSiLU(channels_list[2], channels_list[1], 1, 1)
        self.c2f2 = C2f_FR(channels_list[1] * 2, channels_list[1], 
                          n=max(round(3 * depth_multiple), 1), shortcut=False)
        
        # Bottom-up path (PAN-like)
        self.conv3 = ConvBnSiLU(channels_list[1], channels_list[1], 3, 2)
        self.c2f3 = C2f_FR(channels_list[1] + channels_list[2], channels_list[2], 
                          n=max(round(3 * depth_multiple), 1), shortcut=False)
        
        self.conv4 = ConvBnSiLU(channels_list[2], channels_list[2], 3, 2)
        self.c2f4 = C2f_FR(channels_list[2] + channels_list[3], channels_list[3], 
                          n=max(round(3 * depth_multiple), 1), shortcut=False)
        
    def forward(self, features):
        x2, x3, x4, x5 = features
        
        # Top-down path
        p5 = self.conv1(x5)
        p5_up = self.upsample(p5)
        p4 = torch.cat([p5_up, x4], dim=1)
        p4 = self.c2f1(p4)
        
        p4 = self.conv2(p4)
        p4_up = self.upsample(p4)
        p3 = torch.cat([p4_up, x3], dim=1)
        p3 = self.c2f2(p3)
        
        # Bottom-up path
        p3_down = self.conv3(p3)
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.c2f3(p4)
        
        p4_down = self.conv4(p4)
        p5 = torch.cat([p4_down, p5], dim=1)
        p5 = self.c2f4(p5)
        
        return [p3, p4, p5]
