import torch
import time
from .trainTask import train
from .evalTask import evaluate
import math
import copy
import os
from torch.utils.tensorboard import SummaryWriter


def trainLoop(model, epochs, train_data, val_data, optimizer, criterion,\
     device, bptt, clip_gradient, ntokens, save_model=True, adaptive_dropout=False, logging=False):
    best_val_loss = float('inf')
    best_model = None
    name = time.strftime('state_dict_%Y_%m_%d-%H_%M_%S.pt')
    if logging:
        log_dir = time.strftime('logging/logging_%Y_%m_%d-%H_%M_%S')
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    for epoch in range(1, epochs + 1):
        p  = min(1.5*epoch/100.0,0.2)
        if adaptive_dropout:
            model.set_dropout(p)
            print('dropout in epoch {:2d}: {:.4f}'.format(epoch, p))
        epoch_start_time = time.time()
        model = train(epoch, model, optimizer, criterion,
                      train_data, ntokens, bptt, clip_gradient, device, writer)
        val_loss = evaluate(model, criterion, val_data,
                            ntokens, bptt, device)
        
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.5f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
        if writer:
            print('wriiiiiter')
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ppl', val_ppl, epoch)
            for name, weight in model.named_parameters():
                writer.add_histogram(name,weight, epoch)
                writer.add_histogram(f'{name}.grad',weight.grad, epoch)



        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        if save_model:
            fpath = os.path.join('checkpoints', name)
            torch.save(best_model.state_dict(), fpath)
    model = best_model
    return model


# print('-' * 89)
#     nb_ex = epochs//10
#     steps_per_epoch = len(train_data)//bptt
#     total_steps = epochs*(steps_per_epoch)
#     max_lr = 1e-3
#     gamma = 1
#     for i in range(nb_ex):
#       print('--------------{}---------------'.format(i+1))
#       criterion = nn.CrossEntropyLoss()
#       local_steps = steps_per_epoch*(10+3)
#       opt = torch.optim.RAdam(model.parameters(),\
#           lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
#       # opt = torch.optim.SGD(model.parameters(),\
#       #      lr=0, momentum = 0.9)
#       #optimizer = NoamOpt(model_size=d_model, factor=1, warmup=8000, optimizer=opt)
#       #optimizer = torch.optim.RAdam(model.parameters(), lr=1.6e-6, weight_decay=1e-3)
#       # optimizer = CosineAnnealingWarmupRestarts(opt,\
#       #     first_cycle_steps=int(0.5*total_steps), cycle_mult=1.1, max_lr=max_lr, min_lr=max_lr/25, warmup_steps=int(0.3*total_steps), gamma=0.5)
#       #print('total_steps: {}'.format(len(train_data)//bptt))
#       # optimizer = linearcycleWarmup(optimizer=opt, total_steps=5*len(train_data)//bptt,\
#       #      pct_start=0.8, anneal_strategy='linear', three_phase=True,\
#       #           max_lr=1e-3, steps_per_epoch=len(train_data)//bptt)
#       optimizer = linearcycleWarmup(optimizer=opt, total_steps=local_steps,\
#           pct_start=0.4, anneal_strategy='linear', three_phase=True,\
#               max_lr=max_lr*gamma)
#       gamma *=0.1
      

#       trainLoop(model, 10, train_data, val_data, optimizer,
#                 criterion, device, bptt, clip_grad_norm, ntokens,  save_model=False)

#     test(model, criterion, test_data, ntokens, bptt, device)