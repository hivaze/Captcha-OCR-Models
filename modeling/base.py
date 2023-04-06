import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50, resnet101


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride, padding, dilation=1, pool_ks=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=pool_ks) if pool_ks is not None else None
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.conv(self.bn(x))
        if self.pooling:
            x = self.pooling(x)
        x = self.activation(x)
        return x


class ResNetImageEncoder(nn.Module):
    
    """
    These models accept images with size (64x256).
    It reduces the size of H component to 1, so we can use W component as sequence for RNN.
    Replaces 2 last layers in ResNet with 3 ConvBlocks.
    """
    
    MODELS_DICT = {
        'resnet18': (resnet18, 128),
        'resnet34': (resnet34, 128),
        'resnet50': (resnet50, 512),
        'resnet101': (resnet101, 512)
    }

    def __init__(self, model_name, out_dim=128, dropout=0.1):
        super().__init__()
        self.pre_bath_norm = nn.BatchNorm2d(3)
        cnn_model = self.MODELS_DICT[model_name][0]()
        self.model = nn.Sequential(*(list(cnn_model.children())[:-4]))
        self.head = nn.Sequential(
            ConvBlock(self.MODELS_DICT[model_name][1], self.MODELS_DICT[model_name][1], 3, 1, 1, pool_ks=(2, 1)),
            ConvBlock(self.MODELS_DICT[model_name][1], self.MODELS_DICT[model_name][1], 3, 1, 1, pool_ks=(2, 1)),
            ConvBlock(self.MODELS_DICT[model_name][1], self.MODELS_DICT[model_name][1], 3, 1, 1, pool_ks=(2, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_net = nn.Sequential(
            nn.LayerNorm(self.MODELS_DICT[model_name][1]),
            nn.Linear(self.MODELS_DICT[model_name][1], out_dim)
        )

    def forward(self, x):
        x = self.pre_bath_norm(x)  # [b, 3, h, w]
        x = self.model(x)  # [b, ch, h, w]
        x = self.head(x)
        x = x.flatten(-2)  # [b, ch, h*w]
        x = x.permute(0, 2, 1) # [b, h*w, ch]
        x = self.dropout(x)
        x = self.out_net(x)  # [b, w, out_dim]
        return x


class CNNImageEncoder(nn.Module):
    
    """
    This model accept images with size (64x256).
    It reduces the size of H component to 1, so we can use W component as sequence for RNN.
    """

    def __init__(self, out_dim=128, dropout=0.1):
        super().__init__()
        # self.pre_bath_norm = nn.BatchNorm2d(3)
        self.layers = nn.Sequential(
            ConvBlock(3, 32, 9, 1, 4, pool_ks=2),
            ConvBlock(32, 64, 7, 1, 3, pool_ks=2),
            ConvBlock(64, 128, 5, 1, 2, pool_ks=(2, 1)),
            ConvBlock(128, 128, 3, 1, 1, pool_ks=(2, 1)),
            ConvBlock(128, 128, 3, 1, 1, pool_ks=(2, 1)),
            ConvBlock(128, 128, 3, 1, 1, pool_ks=(2, 1)),
        )  # [b, 128, 1, 64]
        self.dropout = nn.Dropout(dropout)
        self.out_net = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x = self.pre_bath_norm(x)  # [b, 3, h, w]
        x = self.layers(x)  # [b, ch, 2, w]
        x = x.permute(0, 3, 1, 2)  # [b, w, ch, 2]
        x = x.flatten(-2)  # [b, w, 2*ch]
        x = self.dropout(x)
        x = self.out_net(x)  # [b, 64, out_dim]
        return x
