import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    """通道重排（Channel Shuffle）操作"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()  # 交换维度1和2
    x = x.view(batch_size, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    """ShuffleNetV2 基础模块"""
    def __init__(self, in_channels, out_channels, stride, groups=2):
        super().__init__()
        self.stride = stride
        self.groups = groups
        mid_channels = out_channels // 2

        # 如果 stride=2，输入和输出通道数必须匹配
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # stride=1 时，输入通道减半
            assert in_channels == out_channels
            self.branch1 = nn.Identity()  # 直接传递
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x1, x2), dim=1)
        else:
            # 通道分割
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(out, self.groups)
        return out

class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 完整架构"""
    def __init__(self, num_classes=1000, groups=2, width_multiplier=1.0):
        super().__init__()
        self.groups = groups
        self.stage_repeats = [4, 8, 4]  # 各阶段重复次数
        self.stage_out_channels = self._get_stage_channels(width_multiplier)

        # 输入层（标准卷积）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 三个阶段
        self.stage2 = self._make_stage(24, self.stage_out_channels[0], self.stage_repeats[0], stride=2)
        self.stage3 = self._make_stage(self.stage_out_channels[0], self.stage_out_channels[1], self.stage_repeats[1], stride=2)
        self.stage4 = self._make_stage(self.stage_out_channels[1], self.stage_out_channels[2], self.stage_repeats[2], stride=2)

        # 输出层
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _get_stage_channels(self, width_multiplier):
        """根据宽度乘数计算各阶段输出通道数"""
        if width_multiplier == 0.5:
            return [48, 96, 192]
        elif width_multiplier == 1.0:
            return [116, 232, 464]
        elif width_multiplier == 1.5:
            return [176, 352, 704]
        elif width_multiplier == 2.0:
            return [244, 488, 976]
        else:
            raise ValueError("Unsupported width_multiplier")

    def _make_stage(self, in_channels, out_channels, repeats, stride):
        """构建一个阶段的多层 ShuffleUnit"""
        layers = [ShuffleUnit(in_channels, out_channels, stride, self.groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleUnit(out_channels, out_channels, 1, self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)       # (128,128) -> (64,64)
        x = self.maxpool(x)     # (64,64) -> (32,32)
        x = self.stage2(x)      # (32,32) -> (16,16)
        x = self.stage3(x)      # (16,16) -> (8,8)
        x = self.stage4(x)      # (8,8) -> (4,4)
        x = self.conv5(x)       # (4,4) -> (4,4)
        x = self.global_pool(x) # (4,4) -> (1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 示例用法
if __name__ == "__main__":
    model = ShuffleNetV2(num_classes=10, width_multiplier=1.0)  # 10分类任务
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应为 (1, 10)