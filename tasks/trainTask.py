from torch.nn.utils import clip_grad
from utils.util import generate_square_subsequent_mask, get_batch
import time
import torch
import math


def train(epoch, model, optimizer, criterion, train_data,\
     ntokens, bptt, clip_gradient, device, writer=None):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt
    debug_loss = 0
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        curent_index = num_batches*epoch+batch
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        if model.use_aux:
            output, aux_output = model(data, src_mask)
            main_loss = criterion(output.view(-1, ntokens), targets)
            aux_loss = criterion(aux_output.view(-1, ntokens), targets)        
            loss = main_loss*(1-model.aux_weight) + model.aux_weight * aux_loss
            if writer:
                writer.add_scalar('train/main_loss', main_loss.item(), curent_index)
                writer.add_scalar('train/aux_loss', aux_loss.item(), curent_index)
                writer.add_scalar('train/loss', loss.item(), curent_index)
                writer.add_scalar('train/ppl', math.exp(main_loss.item()), curent_index)
                
        else:
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
            if writer:
                writer.add_scalar('train/loss', loss.item(), curent_index)
                writer.add_scalar('train/ppl', math.exp(loss.item()), curent_index)
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()
        total_loss += loss.item()
        if writer:
            writer.add_scalar('lr', optimizer.lr, num_batches*epoch+batch)    
        if batch % log_interval == 0 and batch > 0:
            lr = optimizer.lr
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return model, cur_loss, math.exp(cur_loss)
