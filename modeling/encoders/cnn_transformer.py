import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CNNImageEncoderV2(nn.Module):

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
        )  # [b, 128, 1, 32]
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
        x = self.out_net(x)  # [b, w, out_dim]
        return x


class TransformerImageDecoder(nn.Module):

    def __init__(self, hidden_dim, nhead, dim_feedforward,
                 vocab_size, tr_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.length_embeddings = nn.Embedding(num_embeddings=100, embedding_dim=hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                            dropout=dropout, activation=F.gelu, batch_first=False)
        self.model = nn.TransformerEncoder(layer, tr_layers)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeddings_ids = torch.arange(0, x.shape[1], device=x.device)
        x = self.norm(x)
        x = x + self.length_embeddings(embeddings_ids)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        x = self.model(x, is_causal=False)
        x = self.out_proj(x)
        return x


class OCR_CNNBERT(nn.Module):

    def __init__(self, vocab_size, hidden_dim, nhead, dim_feedforward,
                 tr_layers, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoderV2(out_dim=hidden_dim, dropout=dropout)
        self.decoder = TransformerImageDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, nhead=nhead,
                                               dim_feedforward=dim_feedforward, tr_layers=tr_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x
