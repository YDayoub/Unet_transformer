from models.PositionalEncoder import PositionalEncoding
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import math

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
class UTransformer(nn.Module):

    def __init__(self, ntokens: int,  d_model: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.5, activation: str = 'relu', *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.model_type = 'U-transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead, \
                                                                          dim_feedforward=dim_feedforward,\
                                                                          dropout=dropout, activation=activation)\
                                                   for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, nhead=nhead,\
                                                                          dim_feedforward=dim_feedforward,\
                                                                          dropout=dropout, activation=activation)\
                                                   for _ in range(nlayers)])
        self.embedding = nn.Embedding(ntokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.bottleneck0 =  nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.Dropout(dropout))
        self.act = get_activation_fn(activation)
        self.bottleneck1 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, ntokens)
        self.init_weights()

    def init_weights(self) -> None:
        # def initialization(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #         m.bias.data.fill_(0.01)
        # self.apply(initialization)
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
        #src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.embedding(src) 
        memory = self.pos_encoder(src)
        #memory = self.dropout(memory)
        encoder_outputs = []
        for layer in self.transformer_encoder:
            memory = layer(memory, src_mask=src_mask)
            encoder_outputs.append(memory)
        output = self.norm(self.bottleneck1(self.act(self.bottleneck0(memory)))+memory)
        for layer in self.transformer_decoder:
          output = layer(output, encoder_outputs.pop(),\
                                                 memory_mask=src_mask, tgt_mask=src_mask)              
        output = self.decoder(output)
        return output
