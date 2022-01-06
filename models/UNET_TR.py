import torch
from torch import nn
import numpy as np
'''
This code is adapted from 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
'''

class UnetTransformer(nn.Module):
  '''
    This class implements unet transformer
  '''
  def __init__(self,n_blocks,input_dim,n_heads,d_model,num_classes,dim_feedforward,dropout=0.0):

    '''
      Args:
        n_blocks: number of encoder/decoder blocks
        input_dim: Dimensionality of the input space
        n_heads: number of heads in MultiHeadAttention
        d_model: Dimensionality of the embedding space
        num_classes: Dimensionality of the output space
        dim_feedforward:  Dimensionality of the hidden layer in the MLP 


    '''
    super(UnetTransformer,self).__init__()
    self.n_blocks = n_blocks
    self.pos_enc = Embedding_with_PosEncoding(input_dim,d_model,dropout=dropout)
    self.pos_dec = Embedding_with_PosEncoding(input_dim,d_model,dropout=dropout)
    for i in range(n_blocks):
      vars(self)['_modules']['enc_'+str(i)] = EncoderBlock(d_model, n_heads, dim_feedforward, dropout)
    for i in range(n_blocks):
      vars(self)['_modules']['dec_'+str(i)] = DecoderBlock(d_model, n_heads, dim_feedforward, dropout)
    self.output_layer = nn.Sequential( nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
  def forward(self,x,output_shifted):
    x_encoded = self.pos_enc(x)
    output_decoded = self.pos_dec(output_shifted)
    layers = vars(self)['_modules']
    stack = [x_encoded]
    x = layers['enc_0'](x_encoded)
    for i in range(1,self.n_blocks):
      stack.append(x)
      x = layers['enc_'+str(i)](x)
    stack.append(x)
    x = layers['dec_0'](output_decoded,stack.pop(0))
    for i in range(1,self.n_blocks):
      x = layers['dec_'+str(i)](x,stack.pop(0))
    return self.output_layer(x)


class Embedding_with_PosEncoding(nn.Module):
  def __init__(self,input_dim,d_model, max_len=5000,dropout=0):
    '''
    Args:
      d_model: hidden space dimentionality for Embedding
      input_dim: input space dimentionality
      max_len: maximum length of an input sequence
      drop: probability of an element to be zeroed
    '''
    super(Embedding_with_PosEncoding,self).__init__()
    self.emb = nn.Embedding(input_dim,d_model)
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    # register_buffer is used to save and retrain parameters which don't need to train
    self.register_buffer('pe', pe, persistent=False) 
  def forward(self,x):
    seq_len = x.size(1)
    x = self.emb(x)
    x = self.dropout(x)
    x = x + self.pe[:, :seq_len]
    return x
def scaled_dot_product(query,key,values,mask=None,scale=True):
  '''
      Args:
        query: tensor of queries
        key : tensor of keys
        value: tensor of value
        mask (numpy.ndarray): attention-mask, used to perform self attention when required
        scale (bool): whether to scale the dot product of the query and transposed key
  '''
  if scale:
    depth = query.shape[-1] ** 0.5
  else:
    depth = 1
  dots = torch.matmul(query,torch.swapaxes(key,-1,-2))/depth
  if mask is not None:
    dots = torch.where(mask,dots,torch.full_like(dots, -9e15))
  logsumexp = torch.logsumexp(dots, axis=-1, keepdims=True)
  dots = torch.exp(dots - logsumexp)
  attention = torch.matmul(dots, values)
  return attention

def dot_product_self_attention(q, k, v,device):
  '''
    Args:
        q: queries.
        k: keys.
        v: values.
    Returns:
        masked dot product self attention tensor.  
  '''
  mask_size = q.shape[-2]
  mask = torch.tril(torch.ones((1, mask_size, mask_size), dtype=torch.bool), diagonal=0).to(q.device)        
  return scaled_dot_product(q, k, v, mask)

class QKV(nn.Module):
  '''
  takes as input a tensor of shape (batch_size,seq_len,d_model)
  returns:
  three tensors q,k,v of shape (batch_size,n_heads,seq_len,d_model//n_heads)
  '''

  def __init__(self,n_heads,d_model):
    '''
      Args:
        n_heads: number of heads used in multihead attention
        d_model: hidden space dimensions
    '''
    assert d_model%n_heads==0,'d_models should be divisible by n_heads'
    super(QKV,self).__init__()
    self.qvk = nn.Linear(in_features=d_model,out_features=3*d_model)
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_heads = d_model//n_heads
  def forward(self,x):
    batch_size,seq_len,d_model = x.shape
    x = self.qvk(x)
    x = x.reshape(batch_size,seq_len,self.n_heads,3*self.d_heads)
    x = x.permute(0,2,1,3)
    q,k,v = x.chunk(3,dim=-1)
    return q,k,v 

class MultiheadAttention(nn.Module):
  '''
  This class implements mulithead attention
  '''
  def __init__(self,d_model,causal_attention=False):
    '''
      Args:
        d_model: hidden space dimensions
        causal_attention: boolean whether to use attention or causal attention 
    '''
    super(MultiheadAttention,self).__init__()
    self.d_model = d_model
    self.o = nn.Linear(in_features=d_model,out_features=d_model)
    self.causal_attention = causal_attention 

  def forward(self,q,k,v):
    batch_size,n_heads,seq_len,d_heads = q.shape
    if self.causal_attention:
      atten =  dot_product_self_attention(q, k, v)
    else:
      atten = scaled_dot_product(q,k,v)
    atten = atten.permute(0,2,1,3)
    atten = atten.reshape(batch_size,seq_len,self.d_model)
    res = self.o(atten)
    return res



class EncoderBlock(nn.Module):
  '''
  This class implements encoder block
  '''
  def __init__(self,d_model, n_heads, dim_feedforward, dropout=0.0):
    '''
      Args:
        d_model: hidden space dimensions
        n_heads: number of heads
        dim_feedforward: Dimensionality of the hidden layer in the MLP  
        drop: probability of an element to be zeroed
    '''
    super(EncoderBlock,self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.qkv =  QKV(n_heads=n_heads,d_model=d_model)
    self.attention = MultiheadAttention(d_model=d_model,causal_attention=False)
    self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self,x0):
    q,k,v = self.qkv(x0)
    x1 = self.attention(q,k,v)
    x2 = self.norm1(x0+self.dropout(x1))
    x3 = self.feedforward(x2)
    x4 = self.norm2(self.dropout(x3)+x2)
    return x4



def reshape_tensor(x,n_heads):
  '''
    Args:
      x: tensor of shape (batch_size,seq_len,d_model)
      n_heads: number of heads in mutlihead attention
    Returns:
      reshaped tensor of shape (batch_size,n_heads,seq_len,d_model//n_heads)    
  '''
  batch_size,seq_len,d_model = x.shape
  x = x.reshape(batch_size,seq_len,n_heads,d_model//n_heads)
  x = x.permute(0,2,1,3)
  return x

class DecoderBlock(nn.Module):
  '''
    This class implements decoder block
  '''

  def __init__(self,d_model, n_heads, dim_feedforward, dropout=0.0):
    '''
      Args:
        d_model: hidden space dimensions
        n_heads: number of heads
        dim_feedforward: Dimensionality of the hidden layer in the MLP  
        drop: probability of an element to be zeroed
    '''
    super(DecoderBlock,self).__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.qkv = QKV(n_heads,d_model)
    self.dropout = nn.Dropout(p=dropout)
    self.attention = MultiheadAttention(d_model,causal_attention=False)
    self.causal_attention = MultiheadAttention(d_model,causal_attention=True)
    self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

  def forward(self,x0,skip_con):
    q,k,v = self.qkv(x0)
    x1 = self.causal_attention(q,k,v)
    x2 = self.norm1(x0+self.dropout(x1))
    x3 = reshape_tensor(x2,self.n_heads)
    skip_con = reshape_tensor(skip_con,self.n_heads)
    x4 = self.attention(x3,skip_con,skip_con)
    x5 = self.norm2(x2+self.dropout(x4))
    x6 = self.feedforward(x5)
    x7 = self.norm3(self.dropout(x6)+x5)
    return x7




    