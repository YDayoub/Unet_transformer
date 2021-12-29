import torch
from torch import nn, Tensor
from utils.util import *

'''
Was adpated from 

https://github.com/benkrause/dynamic-evaluation/blob/master/dynamiceval.py

'''

def dynamic_evaluation(model, lr, lamda, data_val,  ntokens, bptt, device):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    total_loss = 0    
    batch, i = 0, 0
    criterion = nn.CrossEntropyLoss()
    for param in model.parameters():
      param.data0 = param.data*1.0
    while i < data_val.size(0)-1-1:
        model.eval()
        data, targets = get_batch(data_val, i, bptt)
        batch_size = data.size(0)
        src_mask = generate_square_subsequent_mask(batch_size).to(device)
        model.zero_grad()
        if model.use_aux:
            output, _, _ = model(data, src_mask)
        else:
            output, _ = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
          for param in model.parameters():
            dW =  lamda*(param.data0-param.data)
            param.data+=dW
        total_loss +=  batch_size * (loss.item())
        batch += (batch_size/bptt)
        i += batch_size
    loss = total_loss/(len(data_val) - 1)
    return loss

