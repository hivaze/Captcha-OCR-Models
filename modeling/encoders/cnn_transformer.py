import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CNNImageEncoder, ResNetImageEncoder


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
        x = self.norm(x + self.length_embeddings(embeddings_ids))
        x = x.permute(1, 0, 2)  # [B, T, in_dim] to [T, B, in_dim]
        x = self.model(x)
        x = self.out_proj(x)
        return x


class OCR_CNNBERT(nn.Module):

    def __init__(self, vocab_size, hidden_dim, nhead, dim_feedforward,
                 tr_layers, dropout=0.1):
        super().__init__()
        self.encoder = CNNImageEncoder(out_dim=hidden_dim, dropout=dropout)
        self.decoder = TransformerImageDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, nhead=nhead,
                                               dim_feedforward=dim_feedforward, tr_layers=tr_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(-1)  # for CTCLoss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x
