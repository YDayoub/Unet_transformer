import torch
import time
from .trainTask import train
from .evalTask import evaluate
import math
import copy
import os


def trainLoop(model, epochs, train_data, val_data, optimizer, criterion,\
     device, bptt, clip_gradient, ntokens, save_model=True, adaptive_dropout=False):
    best_val_loss = float('inf')
    best_model = None
    name = time.strftime('state_dict_%Y_%m_%d-%H_%M_%S.pt')
    for epoch in range(1, epochs + 1):
        p  = min(1.5*epoch/100.0,0.2)
        if adaptive_dropout:
            model.set_dropout(p)
            print('dropout in epoch {:2d}: {:.4f}'.format(epoch, p))
        epoch_start_time = time.time()
        model = train(epoch, model, optimizer, criterion,
                      train_data, ntokens, bptt, clip_gradient, device)
        val_loss = evaluate(model, criterion, val_data,
                            ntokens, bptt, device)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.5f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        if save_model:
            fpath = os.path.join('checkpoints', name)
            torch.save(best_model.state_dict(), fpath)
