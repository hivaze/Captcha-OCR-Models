import torch.nn as nn
from transformers import ViTConfig, ViTModel


class BiLSTMImageDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, vocab_size, lstm_layers, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(65)
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, bidirectional=True,
                           batch_first=False)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        out, _ = self.rnn(x)
        x = self.out_proj(out)
        return x


class OCR_ViTRNN(nn.Module):

    def __init__(self, vit_config: ViTConfig, vocab_size: int, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.encoder = ViTModel(vit_config, add_pooling_layer=False)
        self.decoder = BiLSTMImageDecoder(in_dim=vit_config.hidden_size, hidden_dim=vit_config.hidden_size,
                                          vocab_size=vocab_size, lstm_layers=lstm_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.encoder(x).last_hidden_state
        x = self.decoder(x)
        x = self.softmax(x)
        return x


class OCR_ViT(nn.Module):

    def __init__(self, vit_config: ViTConfig, vocab_size: int):
        super().__init__()
        self.model = ViTModel(vit_config, add_pooling_layer=False)
        self.out_proj = nn.Linear(vit_config.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = x.permute(1, 0, 2)
        x = self.out_proj(x)
        x = self.softmax(x)
        return x