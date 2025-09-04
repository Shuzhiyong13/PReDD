import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    """Conv + BN + ReLU6 """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """Inverted Residual Block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))
       
        layers.extend([
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
          
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super().__init__()
        input_channels = 32
        last_channels = 1280

        inverted_residual_setting = [
            # t, c,  n, s
            [1, 16,  1, 1],
            [6, 24,  2, 2],
            [6, 32,  3, 2],
            [6, 64,  4, 2],
            [6, 96,  3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]


        input_channels = int(input_channels * width_multiplier)
        self.last_channels = int(last_channels * width_multiplier) if width_multiplier > 1.0 else last_channels
        features = [ConvBNReLU(3, input_channels, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channels = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, output_channels, stride, t))
                input_channels = output_channels

        features.append(ConvBNReLU(input_channels, self.last_channels, kernel_size=1))
        self.features = nn.Sequential(*features)


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.last_channels, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = MobileNetV2(num_classes=10)  # 10 
    dummy_input = torch.randn(1, 3, 128, 128)  # (batch, channels, height, width)
    output = model(dummy_input)
    print("Output shape:", output.shape)  #  (1, 10)