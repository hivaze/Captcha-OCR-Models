import torch
import torch.nn as nn

from ..base import ResNetImageEncoder


class BiLSTMImageDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, vocab_size, lstm_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True,
                           batch_first=False)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        rnn_out, _ = self.rnn(x)
        x = self.out_proj(rnn_out)
        return x


class OCR_ResNetRNN(nn.Module):

    def __init__(self, resnet_model, vocab_size, hidden_dim=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = ResNetImageEncoder(resnet_model, out_dim=hidden_dim, dropout=dropout)
        self.decoder = BiLSTMImageDecoder(in_dim=hidden_dim, hidden_dim=hidden_dim,
                                          vocab_size=vocab_size, lstm_layers=lstm_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x