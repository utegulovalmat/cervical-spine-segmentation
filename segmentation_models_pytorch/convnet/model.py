"""
history:
v1:
CBR filters: (conv64-bn-relu) (conv128-bn-relu) (conv128-1) sigmoid

v2:
CBR filters: (conv32-bn-relu) (conv64-bn-relu) (conv128-bn-relu)
(conv256-bn-relu) (conv512-bn-relu) (conv512x1) sigmoid

v3:
=== 5 down - 5 up
CBR-Large32-512 filters: (conv32-bn-relu) maxpool (conv64-bn-relu) maxpool
(conv128-bn-relu) maxpool (conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
... sigmoid

=== 4 down - 4 up
CBR-Large64-512 filters: (conv64-bn-relu) maxpool (conv128-bn-relu) maxpool
(conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
... sigmoid

CBR-Small32-256 filters: (conv32-bn-relu) maxpool (conv64-bn-relu) maxpool
(conv128-bn-relu) maxpool (conv256-bn-relu) maxpool
... sigmoid

CBR-Small64-512 filters: (conv64-bn-relu) maxpool (conv128-bn-relu) maxpool
(conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
... sigmoid
"""
import torch
import torch.nn as nn


class ConvBnReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        use_batchnorm=True,
    ):
        super().__init__()
        interim_channels = out_channels * 2
        conv = nn.Conv2d(
            in_channels,
            interim_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)
        conv11 = nn.Conv2d(
            interim_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=not use_batchnorm,
        )
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(ConvBnReLU, self).__init__(conv, relu, conv11, bn, relu)


class ConvNet(nn.Module):
    def __init__(
        self, size: str = "small64", in_channels: int = 1, classes: int = 1,
    ):
        super(ConvNet, self).__init__()
        """
        """
        self.size = size
        self.conv64 = ConvBnReLU(in_channels, 64)
        self.conv128 = ConvBnReLU(64, 128)
        self.conv256 = ConvBnReLU(128, 256)
        self.last = nn.Conv2d(256, classes, kernel_size=1, stride=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv64(x)
        x = self.maxpool(x)
        x = self.conv128(x)
        x = self.maxpool(x)
        x = self.conv256(x)
        x = self.up(x)
        x = self.last(x)
        out = self.sigmoid(x)
        return out

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
