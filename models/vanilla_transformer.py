import math
from torch import nn, Tensor
from models.PositionalEncoder import PositionalEncoding
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class VanillaTransformer(nn.Module):

    def __init__(self, ntokens: int,  d_model: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.5,\
                      activation: str = 'relu', *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.model_type = 'Vanilla Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=dropout, activation=activation)
                                                  for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=dropout, activation=activation)
                                                  for _ in range(nlayers)])
        self.embedding = nn.Embedding(ntokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor,  src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        memory = self.pos_encoder(src)
        memory = self.dropout(memory)
        for layer in self.transformer_encoder:
            memory = layer(memory, src_mask=src_mask)
        output = memory
        for layer in self.transformer_decoder:
            output = layer(output, memory,
                           memory_mask=src_mask, tgt_mask=src_mask)
        output = self.decoder(output)
        return output
