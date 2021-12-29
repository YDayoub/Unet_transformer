import torch
import time
from .trainTask import train
from .evalTask import evaluate
import math
import copy
import os
from torch.utils.tensorboard import SummaryWriter
from utils.util import save_model

def trainLoop(model, config, start_epoch, epochs, train_data, val_data, optimizer, criterion,\
     device, bptt, clip_gradient, ntokens, alpha, beta, save_model_flag=True,\
          adaptive_dropout=False, logging=False, log_dir=None,use_var_len=False,\
              partial_shuffling=False,use_average=False, custom_loss=None):
    best_val_loss = float('inf')
    best_model = None
    name = time.strftime('{}_state_dict_%Y_%m_%d-%H_%M_%S.pth'.format(model.model_type))
    writer = SummaryWriter(log_dir=log_dir) if logging else None
    for epoch in range(start_epoch, epochs + 1):
        p  = min(1.5*epoch/100.0,0.4)
        if adaptive_dropout:
            with torch.no_grad():
                model.set_dropout(p)
            print('dropout in epoch {:2d}: {:.4f}'.format(epoch, p))
        epoch_start_time = time.time()
        train_criterion =  custom_loss if custom_loss else criterion
        model, train_loss, train_ppl = train(epoch, model, optimizer, train_criterion,
                      train_data, ntokens=ntokens, bptt=bptt, clip_gradient=clip_gradient, device=device, writer=writer,\
                          use_var_len=use_var_len,alpha = alpha, beta=beta,\
                              partial_shuffling=partial_shuffling, use_average=use_average)
        val_loss = evaluate(model, criterion, val_data,
                            ntokens, bptt, device)
        optimizer.schedule_step(val_loss)
        
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.6f} | valid ppl {val_ppl:8.6f}')
        print('-' * 89)
        if writer:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ppl', val_ppl, epoch)
            #writer.add_histogram('decoder/weights', model.decoder.weight, epoch)
            #writer.add_histogram('decoder/bias', model.decoder.bias, epoch)
            # for name, weight in model.named_parameters():
            #     writer.add_histogram(name,weight, epoch)
            #     writer.add_histogram(f'{name}.grad',weight.grad, epoch)



        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        if save_model_flag:
            fpath = os.path.join('checkpoints', name)
            checkpoints = {
                'model': best_model,
                'configs': config,
                'epoch': epoch,
                'optimizer': optimizer,
            }
            save_model(checkpoints, fpath)
    model = best_model
    val_loss = evaluate(model, criterion, val_data,
                            ntokens, bptt, device)
    val_ppl = math.exp(val_loss)


    return model, train_loss, train_ppl, val_loss, val_ppl


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