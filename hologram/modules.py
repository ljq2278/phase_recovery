import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, channels=16):
        super(ResidualBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        output = x + self.double_conv(x)
        return output


class UpsampleBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, channels=16, mid_channel=64):
        super(UpsampleBlock, self).__init__()
        self.inflate_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channel, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(mid_channel, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        output = self.inflate_conv(x)
        return output
