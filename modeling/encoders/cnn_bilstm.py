import torch
import torch.nn as nn


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


class CNNImageEncoder(nn.Module):

    def __init__(self, max_seq_length=32, out_dim=128, dropout=0.1):
        super().__init__()
        # self.pre_bath_norm = nn.BatchNorm2d(3)
        self.layers = nn.Sequential(
            ConvBlock(3, 32, 11, 1, 5),
            ConvBlock(32, 64, 9, 1, 4, pool_ks=2),
            ConvBlock(64, 64, 7, 1, 3, pool_ks=2),
            ConvBlock(64, 128, 5, 1, 2, pool_ks=(2, 1)),
            ConvBlock(128, 128, 5, 1, 2, pool_ks=(2, 1)),
            ConvBlock(128, 128, 3, 1, 1, pool_ks=(2, 1)),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((max_seq_length, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.out_net = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x = self.pre_bath_norm(x)  # [b, 3, h, w]
        x = self.layers(x)  # [b, ch, h, w]
        x = x.permute(0, 3, 1, 2)  # [b, w, ch, h]
        x = x.flatten(-2)  # [b, w, ch*h]
        x = self.dropout(x)
        x = self.avg_pool(x)  # [b, max_length, out_dim]
        x = self.out_net(x)  # [b, w, out_dim]
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


class BiLSTMImageDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, vocab_size, lstm_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(64)
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True,
                           batch_first=False)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        rnn_out, _ = self.rnn(x)
        x = self.out_proj(rnn_out)
        return x


class OCR_CRNN(nn.Module):

    def __init__(self, vocab_size, hidden_dim=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoderV2(out_dim=hidden_dim, dropout=dropout)
        self.decoder = BiLSTMImageDecoder(in_dim=hidden_dim, hidden_dim=hidden_dim,
                                          vocab_size=vocab_size, lstm_layers=lstm_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x


class SelfAttenBiLSTMImageDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, vocab_size, num_heads, lstm_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(64)
        self.self_atten = nn.MultiheadAttention(in_dim, num_heads, dropout, batch_first=False)
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True,
                           batch_first=False)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        attn_output, attn_output_weights = self.self_atten(x, x, x)
        rnn_out, _ = self.rnn(attn_output)
        x = self.out_proj(rnn_out)
        return x


class OCR_CARNN(nn.Module):

    def __init__(self, vocab_size, hidden_dim=128, num_heads=2, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoderV2(out_dim=hidden_dim, dropout=dropout)
        self.decoder = SelfAttenBiLSTMImageDecoder(in_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                                   vocab_size=vocab_size, lstm_layers=lstm_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x


class CrossAttenBiLSTMImageDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, vocab_size, num_heads, lstm_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(64)
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True,
                           batch_first=False)
        self.cross_atten = nn.MultiheadAttention(embed_dim=hidden_dim * 2, kdim=in_dim,
                                                 vdim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        rnn_out, _ = self.rnn(x)
        attn_output, attn_output_weights = self.cross_atten(rnn_out, x, x)
        x = self.out_proj(attn_output)
        return x


class OCR_CRNNA(nn.Module):

    def __init__(self, vocab_size, hidden_dim=128, num_heads=2, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoderV2(out_dim=hidden_dim, dropout=dropout)
        self.decoder = CrossAttenBiLSTMImageDecoder(in_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                                    vocab_size=vocab_size, lstm_layers=lstm_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x
