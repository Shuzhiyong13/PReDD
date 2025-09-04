import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish(),
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                      padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
        )
        
        # Squeeze-and-Excitation (SE) block
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid(),
        )
        
        # Output phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        residual = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x) * x  # SE block
        x = self.project(x)
        
        if self.use_residual:
            x += residual
        
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0, depth_multiplier=1.0):
        super().__init__()
        
        # Base configuration for EfficientNet-B0
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_repeats = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]
        
        # Scale channels and repeats
        channels = [int(c * width_multiplier) for c in base_channels]
        repeats = [int(r * depth_multiplier) for r in base_repeats]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish(),
        )
        
        # MBConv blocks
        blocks = []
        in_channels = channels[0]
        for i in range(7):
            out_channels = channels[i+1]
            for j in range(repeats[i]):
                stride = strides[i] if j == 0 else 1
                blocks.append(MBConvBlock(in_channels, out_channels, kernel_sizes[i], stride, expand_ratios[i]))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels[-1], 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
    # Example usage
if __name__ == "__main__":
    model = EfficientNet(num_classes=10)  # For 10-class classification
    dummy_input = torch.randn(1, 3, 128, 128)  # (batch, channels, height, width)
    output = model(dummy_input)
    print("Output shape:", output.shape) 