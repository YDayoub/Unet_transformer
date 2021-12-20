from torch.nn.modules.loss import _WeightedLoss
from models.PositionalEncoder import PositionalEncoding
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import math
from models.embed_regularize import embedded_dropout

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class UTransformer(nn.Module):

    def __init__(self, ntokens: int,  d_model: int, nhead: int, dim_feedforward: int,
                 nlayers: int, drop_rate: float = 0.4,in_dropout: float=0.65, emb_dropout: float=0.1, out_dropout: float=0.4, activation: str = 'relu',\
                      use_aux = False, weight=None, tying=False, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.model_type = 'U-transformer'
        self.pos_encoder = PositionalEncoding(d_model, in_dropout)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead, \
                                                                          dim_feedforward=dim_feedforward,\
                                                                          dropout=drop_rate, activation=activation, norm_first=True)\
                                                   for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, nhead=nhead,\
                                                                          dim_feedforward=dim_feedforward,\
                                                                          dropout=drop_rate, activation=activation, norm_first=True)\
                                                   for _ in range(nlayers)])
        self.embedding = nn.Embedding(ntokens, d_model, scale_grad_by_freq=False)
        self.dropout = nn.Dropout(out_dropout)
        self.emb_dropout = emb_dropout

        self.d_model = d_model
        self.tying = tying
        self.decoder = nn.Linear(d_model, ntokens)
        self.drop_out = nn.Dropout(0)
        if self.tying:
            self.decoder.weight = self.embedding.weight
        self.use_aux = use_aux
        self.dropout_val = drop_rate
        if self.use_aux:
            print('-----------using_aux_output----------------')
            self.aux_weight = weight
            self.decoder_aux = nn.Linear(d_model, ntokens)
        self.init_weights()
        # self.bottleneck0 =  nn.Sequential(
        #     nn.Linear(d_model, d_model), nn.Dropout(dropout))
        # self.act = get_activation_fn(activation)
        # self.bottleneck1 = nn.Linear(d_model, d_model)
        #self.decoder = nn.Linear(d_model, ntokens)

        #self.init_weights()

    def init_weights(self) -> None:
        def initialization(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.01)
        self.apply(initialization)
        # initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        # if not self.tying:
        #     self.decoder.bias.data.zero_()
        #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def set_dropout(self, drop_rate=0.1)-> None:
        def set_dropout_rec(model, p):
            for _, child in model.named_children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = drop_rate
                set_dropout_rec(child, p)
        set_dropout_rec(self, drop_rate)

    def forward(self, src: Tensor,  src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        #src = self.embedding(src) * math.sqrt(self.d_model)
        #src = self.embedding(src) 
        src = embedded_dropout(self.embedding, src,\
            dropout=self.emb_dropout)
        memory = self.pos_encoder(src)
        encoder_outputs = []
        hidden_states = []
        for layer in self.transformer_encoder:
            memory = layer(memory, src_mask=src_mask)
            hidden_states.append(memory)
            encoder_outputs.append(memory)
        output = memory
        if self.use_aux:
            aux_output = self.decoder_aux(output)
        for layer in self.transformer_decoder:
            output = layer(output, encoder_outputs.pop(),\
                        memory_mask=src_mask, tgt_mask=src_mask)  
                                                 
            hidden_states.append(output)
        
        ####### outputs ######
        
        output = self.drop_out(output)
  
        if self.tying:
            output = F.linear(output, self.decoder.weight, self.decoder.bias)
        else:      
            output = self.decoder(output)
        return (output, aux_output, hidden_states) if self.use_aux else (output, hidden_states)
