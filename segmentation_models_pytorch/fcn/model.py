from typing import Optional, Union, List

import torch
import torch.nn as nn
from torchvision import models


class FCN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        # encoder_weights: str = "imagenet",
        # decoder_use_batchnorm: bool = True,
        # decoder_channels: List[int] = (256, 128, 64, 32, 16),
        # decoder_attention_type: Optional[str] = None,
        # in_channels: int = 3,
        classes: int = 1,
        # activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
    ):
        super(FCN, self).__init__()
        if encoder_name == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
        else:
            raise
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode="bilinear")

        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)

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
