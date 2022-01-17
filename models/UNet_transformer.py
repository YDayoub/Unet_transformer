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


def squared_relu(x):
    res = F.relu(x)
    return res*res

class UTransformer(nn.Module):

    def __init__(self, ntokens: int,  d_model: int, nhead: int, dim_feedforward: int,
                 nlayers: int, drop_rate: float = 0.4, in_dropout: float = 0.65, emb_dropout: float = 0.1, out_dropout: float = 0.4, activation: callable = torch.nn.functional.relu,
                 use_aux=False, weight=None, tying=False, mos: bool = True, n_experts: int = 3,
                 save_state: bool = False, adv_tr: bool = False, epsilon: float = 0.002,
                 gaussian: float = 0.2, weighted_connections=False, use_gru=False, ar_tar=False,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'U-transformer'
        self.pos_encoder = PositionalEncoding(d_model, in_dropout)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=drop_rate, activation=squared_relu)
                                                  for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=drop_rate, activation=squared_relu)
                                                  for _ in range(nlayers)])
        self.embedding = nn.Embedding(ntokens, d_model)
        self.dropout = nn.Dropout(out_dropout)
        self.emb_dropout = emb_dropout
        self.d_model = d_model
        self.tying = tying
        self.ntokens = ntokens
        self.mos = mos
        self.n_experts = n_experts
        self.save_state = save_state
        self.adv_tr = adv_tr
        self.weighted_connections = weighted_connections
        self.epsilon = epsilon
        self.gaussian = gaussian
        self.use_gru = use_gru
        self.decoder = nn.Linear(d_model, ntokens)
        if weighted_connections:
            self.skip_weights = nn.Parameter(torch.diag(torch.ones(nlayers)))
        if self.tying:
            self.decoder.weight = self.embedding.weight
        if self.mos:
            self.prior = nn.Linear(d_model, n_experts, bias=False)
            self.latent = nn.Sequential(nn.Linear(d_model, n_experts*d_model),
                                        nn.Tanh())

        if self.use_gru:
            self.gru = nn.GRU(input_size=2*d_model, hidden_size =d_model, batch_first =False,\
                dropout=0.1)


        self.use_aux = use_aux
        self.dropout_val = drop_rate
        if self.use_aux:
            print('-----------using_aux_output----------------')
            self.aux_weight = weight
            self.decoder_aux = nn.Linear(d_model, ntokens)
        self.init_weights()

    def init_weights(self) -> None:
        def initialization(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                try:
                    m.bias.data.fill_(0.01)
                except Exception as e:
                    pass
        self.apply(initialization)

    def set_dropout(self, drop_rate=0.1) -> None:
        def set_dropout_rec(model, p):
            for _, child in model.named_children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = drop_rate
                set_dropout_rec(child, p)
        set_dropout_rec(self, drop_rate)

    def forward(self, src: Tensor,  src_mask: Tensor, h=None, gru_h=None, targets=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        #src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.embedding(src)
        if self.training and self.adv_tr:
            m = torch.distributions.Normal(
                torch.zeros_like(src), torch.ones_like(src) * 1.)
            sigma = m.sample() * self.gaussian
            src = src + sigma
        if self.use_gru:
            first_hidden, last_hiddens = torch.split(src,[1,src.shape[0]-1])         
            h0 = torch.cat([first_hidden, h], dim=-1)
            h0, gru_h = self.gru(h0,gru_h)
            src = torch.cat([h0, last_hiddens],dim=0)

            
        memory = self.pos_encoder(src)

        encoder_outputs = []
        hidden_states = []
        for layer_idx, layer in enumerate(self.transformer_encoder):
            memory = layer(memory, src_mask=src_mask)
            hidden_states.append(memory)                
            encoder_outputs.append(memory)
        states = torch.stack(encoder_outputs)
        if self.save_state:
            output = h
            new_h = memory.detach()
        else:
            output = memory
        if self.use_aux:
            aux_output = self.decoder_aux(output)
        
        if self.weighted_connections:
            skip_weights = self.skip_weights.softmax(dim = -1) # nlayers, nlayers


        for layer_id, layer in enumerate(self.transformer_decoder):
            if self.weighted_connections:
                weight = skip_weights[layer_id].view(-1,1,1,1)
                current_state = (weight*states).sum(0)
            else:
                current_state = encoder_outputs.pop()
            output = layer(output, current_state,
                           memory_mask=src_mask, tgt_mask=src_mask)
            # output = layer(output, encoder_outputs.pop(),
            #                memory_mask=src_mask, tgt_mask=src_mask)

            hidden_states.append(output)
        if self.use_gru:
            new_h = output[-1,:,:].detach().view(1,-1,self.d_model)
            #print('new_h {}'.format(new_h.shape))


        output = self.dropout(output)

        ####### outputs ######
        if self.mos:
            shape = output.shape
            prior = self.prior(output).contiguous()
            prior = nn.functional.softmax(prior, -1)
            latent = self.latent(output).view(
                shape[0], shape[1], self.n_experts, -1).contiguous()
        else:
            latent = output
        if self.adv_tr and self.training:
            logits = self.decoder(latent.view(-1, self.d_model))
            _latent = latent.view(-1, self.d_model)
            targets = targets.view(
                [-1, 1]).expand([-1, self.n_experts]).contiguous().view(-1)
            weight_noise = torch.zeros_like(self.decoder.weight).cuda()
            neg_h = -_latent / \
                torch.sqrt(torch.sum(_latent**2, 1, keepdim=True) + 1e-8)
            n_output = torch.sqrt(
                torch.sum(_latent**2, 1, keepdim=True) + 1e-8)
            n_w = torch.sqrt(torch.sum(self.embedding(
                targets)**2, 1, keepdim=True) + 1e-8)
            cos_theta = (torch.sum(_latent * self.embedding(targets),
                         1, keepdim=True)) / n_output / n_w
            indicator = torch.gt(cos_theta, 0e-1).view(-1,
                                                       1).type(torch.cuda.FloatTensor)
            sigma = self.epsilon * n_w * indicator
            weight_noise[targets.view(-1)] = sigma.detach() * neg_h.detach()
            noise_outputs = (_latent * weight_noise[targets]).sum(1)
            logits[torch.arange(targets.size(0)).long().cuda(),
                   targets] += noise_outputs
        else:
            logits = self.decoder(latent)
        if self.mos:
            prob = nn.functional.softmax(
                logits, -1).view(shape[0], shape[1], self.n_experts, self.ntokens)
            output = torch.einsum('ijk,ijkl->ijl', prior, prob)
            output = torch.log(output.add_(1e-8))
        else:
            output = logits
        outputs = [output, hidden_states]
        if self.use_aux:
            outputs = outputs + [aux_output]
        if self.use_gru:
            outputs = outputs+[[new_h,gru_h.detach()]]
        return outputs
