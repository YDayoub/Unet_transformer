from torch.nn.utils import clip_grad
from utils.util import generate_square_subsequent_mask, get_batch
import time
import torch
import math
import numpy as np
import torch.nn as nn
import gc
from models.EMA import ema
from .partial_shuffling import partial_shuffle


def get_sequence_length(bptt, use_var_length):
    if not use_var_length:
        return bptt
    seq_len = bptt if np.random.random() < 0.95 else bptt // 2
    seq_len = max(5, int(np.random.normal(seq_len, 5)))
    seq_len = min(seq_len, bptt + 30)
    return seq_len


def train(epoch, model, optimizer, criterion, train_data,
          ntokens, bptt, clip_gradient, device, alpha, beta, use_average, partial_shuffling, \
              printing=True, writer=None, use_var_len=False):
    model.train()  # turn on train mode
    total_loss = 0.
    total_loss_main = 0.
    log_interval = 200
    start_time = time.time()
    num_batches = len(train_data) // bptt

    i, batch = 0, 0
    if use_average:
        ema_model = None
    if partial_shuffling:
        print('parital shuffing')
        train_data = partial_shuffle(train_data)
    prev_h = None
    gru_h = None

    while i < train_data.size(0) - 1 - 1:
        data, targets = get_batch(train_data, i, get_sequence_length(bptt, use_var_len))
        curent_index = num_batches*epoch+batch
        seq_length = data.size(0)
        # if prev_h is None and model.save_state:
        if prev_h is None and model.use_gru:
            shape = data.shape
            prev_h = torch.zeros(
                (1, shape[1], model.d_model), device='cuda')
            gru_h = torch.zeros(
                (1, shape[1], model.d_model), device='cuda')
        # elif prev_h is not None and prev_h.shape[0] != seq_length:
        #     prev_h = prev_h[-seq_length:,:,:]

        src_mask = generate_square_subsequent_mask(seq_length).to(device)
        outputs = model(data, src_mask, prev_h, gru_h=gru_h, targets=targets)
        output = outputs[0] 
        if alpha or beta:
            hidden_states = outputs[1]
        main_loss = criterion(output.view(-1, ntokens), targets)
        loss = 1.0*main_loss
        if model.use_aux:
            aux_output = outputs[2]
            aux_loss = criterion(aux_output.view(-1, ntokens), data.view(-1))
            loss += model.aux_weight * aux_loss
        # if model.save_state:
        if model.use_gru:
            prev_h = outputs[-1][0]
            gru_h = outputs[-1][1]
            #print('prev_h {}'.format(prev_h.shape))

        if alpha:
            ar_loss = sum(alpha * h.pow(2).mean() for h in hidden_states)
            loss += ar_loss
        if beta:
            tar_loss = sum(beta*(h[1:]-h[:-1]).pow(2).mean()
                           for h in hidden_states)
            loss += tar_loss

        scale_lr = seq_length/bptt
        optimizer.scalar = scale_lr

        optimizer.zero_grad()
        loss.backward()
        if clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()

        total_loss_main += main_loss.item()
        total_loss += loss.item()
        if model.use_aux and writer:
            writer.add_scalars('train/loss', {'main_loss': main_loss.item(),
                                              'aux_loss': aux_loss.item(), 'loss': loss.item()}, curent_index)
            writer.add_scalar('lr', optimizer.lr, curent_index)
            writer.add_scalars('train/ppl', {'train_ppl':  math.exp(main_loss.item(
            )), 'lr': optimizer.lr, 'dropout': model.dropout_val}, curent_index)

        elif writer:
            writer.add_scalar('train/lr', optimizer.lr, curent_index)
            writer.add_scalar('train/loss', loss.item(), curent_index)
            writer.add_scalars('train/ppl', {'train_ppl/100':  math.exp(loss.item())/100,
                                             'lr': optimizer.lr, 'dropout': model.dropout_val}, curent_index)

        if batch % log_interval == 0 and batch > 0:

            lr = optimizer.lr
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_loss_main = total_loss_main / log_interval
            ppl = math.exp(cur_loss_main)
            if printing:
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.8f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.6f} |loss_main {cur_loss_main:5.6f} | ppl {ppl:8.6f}| seq_len {seq_length:5d}')
            total_loss = 0.0
            total_loss_main = 0.0
            start_time = time.time()
            if use_average:
                if ema_model is None:
                    ema_model = ema(model, decay_rate=0.9)
                else:
                    ema_model(model)
        batch += 1
        i += seq_length

    # if use_average:
    #     ema_model.set_model_to_ema(model)
    #     del ema_model
    return model, cur_loss, math.exp(cur_loss)
