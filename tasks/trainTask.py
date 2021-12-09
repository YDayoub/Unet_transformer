from torch.nn.utils import clip_grad
from utils.util import generate_square_subsequent_mask, get_batch
import time
import torch
import math
import numpy as np

def get_sequence_length(bptt, use_var_length):
    if not use_var_length:
        return bptt
    seq_len = bptt if np.random.random() < 0.95 else bptt // 2
    seq_len = max(5, int(np.random.normal(seq_len, 10))) 
    seq_len = min(seq_len, bptt + 30)
    return seq_len
    

def train(epoch, model, optimizer, criterion, train_data,\
     ntokens, bptt, clip_gradient, device, writer=None, use_var_len=False):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    num_batches = len(train_data) // bptt

    hist_counter = 0
    i, batch =0, 0
    
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    while i < train_data.size(0) - 1 - 1:
        seq_len = get_sequence_length(bptt, use_var_len)
        data, targets = get_batch(train_data, i, seq_len)
        curent_index = num_batches*epoch+batch
        batch_size = data.size(0)
        src_mask = generate_square_subsequent_mask(batch_size).to(device)
        # if batch_size != bptt:  # only on last batch
        #     src_mask = src_mask[:batch_size, :batch_size]
        if model.use_aux:
            output, aux_output = model(data, src_mask)
            main_loss = criterion(output.view(-1, ntokens), targets)
            aux_loss = criterion(aux_output.view(-1, ntokens), targets)        
            loss = main_loss*(1-model.aux_weight) + model.aux_weight * aux_loss                
        else:
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        if (batch % log_interval)== 0 and batch > 0:
            if writer:
                step = (num_batches//log_interval)*epoch+hist_counter
                hist_counter += 1
                #writer.add_histogram('decoder/weights.grad', model.decoder.weight.grad, step)
                #writer.add_histogram('decoder/bias.grad', model.decoder.bias.grad, step)
        
        if clip_gradient>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()

        total_loss += loss.item()
        if model.use_aux and writer:
            writer.add_scalars('train/loss', {'main_loss': main_loss.item(),\
                'aux_loss': aux_loss.item(), 'loss': loss.item()}, curent_index)
            writer.add_scalar('lr', optimizer.lr, curent_index)
            writer.add_scalars('train/ppl', {'train_ppl/100':  math.exp(main_loss.item())/100, 'lr': optimizer.lr, 'dropout': model.dropout_val}, curent_index)

        elif writer:
            writer.add_scalar('train/lr', optimizer.lr, curent_index)
            writer.add_scalar('train/loss', loss.item(), curent_index)
            writer.add_scalars('train/ppl', {'train_ppl/100':  math.exp(loss.item())/100,\
                             'lr': optimizer.lr, 'dropout': model.dropout_val}, curent_index)


  
        if batch % log_interval == 0 and batch > 0:
            lr = optimizer.lr
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}| seq_len {batch_size:5d}')
            total_loss = 0
            start_time = time.time()
        batch +=1
        i += seq_len
    return model, cur_loss, math.exp(cur_loss)
