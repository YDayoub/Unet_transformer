import math
import torch
from torch import nn, Tensor
from models.PositionalEncoder import PositionalEncoding
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from models.embed_regularize import embedded_dropout

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class VanillaTransformer(nn.Module):

    def __init__(self, ntokens: int,  d_model: int, nhead: int, dim_feedforward: int,
                 nlayers: int, drop_rate: float = 0.5,\
                      emb_dropout: float = 0.1, activation: str = 'relu', use_aux = False, weight=None, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.model_type = 'Vanilla Transformer'
        self.pos_encoder = PositionalEncoding(d_model, drop_rate)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=drop_rate, activation=activation)
                                                  for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=drop_rate, activation=activation)
                                                  for _ in range(nlayers)])
        self.embedding = nn.Embedding(ntokens, d_model, scale_grad_by_freq=True)
        self.dropout = nn.Dropout(drop_rate)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)
        self.use_aux = use_aux
        self.emb_dropout = emb_dropout
        self.dropout_val = drop_rate
        if self.use_aux:
            self.aux_weight = weight
            self.decoder_aux = nn.Linear(d_model, ntokens)
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
    def set_dropout(self, drop_rate=0.1)-> None:
        def set_dropout_rec(model, p):
            for _, child in model.named_children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = drop_rate
                set_dropout_rec(child, p)
        self.dropout_val = drop_rate
        set_dropout_rec(self, drop_rate)

    def forward(self, src: Tensor,  src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
            
        src = embedded_dropout(self.embedding,src, \
            dropout=self.emb_dropout)*math.sqrt(self.d_model) 
        memory = self.pos_encoder(src)
        #memory = embedded_dropout(self.embedding,src)
        for layer in self.transformer_encoder:
            memory = layer(memory, src_mask=src_mask)
        output = memory
        if self.use_aux:
            aux_output = self.decoder_aux(output)
        for layer in self.transformer_decoder:
          output = layer(output, memory,\
                                                 memory_mask=src_mask, tgt_mask=src_mask)              
        output = self.decoder(output)
        #output = torch.matmul(output, self.embedding.weight.t())
        return (output, aux_output) if self.use_aux else output
