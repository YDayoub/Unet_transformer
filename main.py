import torch
from torch import nn, Tensor
import torch.nn.functional as F
from optimizers.NoamOptimizer import NoamOpt
from optimizers.cos_sch_warmupR import CosineAnnealingWarmupRestarts
from optimizers.linear_warmup import linearcycleWarmup
from models.UNet_transformer import UTransformer
from models.vanilla_transformer import VanillaTransformer
from datasets.WIKI103 import wiki103
from datasets.WIKI2 import wiki2
from tasks.train_loop import trainLoop
from utils.util import *
from torchtext.data.utils import get_tokenizer
from functools import partial
from tasks.testTask import test
import argparse
from tasks.csv_logger import log_data
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='path to the config file', required=True)
    try:
        args = parser.parse_args()
        config_path = args.config
        config = load_config(config_path)
    except Exception as e:
        print('Error', e)
        exit(0)


    #--------------- Reproducibility -------------#
    set_seed(42)

    model_config = config['model_config']
    training_config = config['training']
    eval_config = config['eval']
    dataset_config = config['dataset_config']
    #---------------loadconfig--------------------#
    train_batch_size = training_config['batch_size']
    eval_batch_size = eval_config['batch_size']
    epochs = training_config['n_epochs']
    bptt = training_config['bptt']
    use_aux = training_config['use_aux']
    weight_aux = training_config['weight_aux']
    # gradient norm clipping value
    clip_grad_norm = training_config['clip_grad_norm']
    d_model = model_config['d_model']  # embedding dimension
    # dimension of the feedforward network model in nn.TransformerEncoder
    dim_feedforward = model_config['dim_feedforward']
    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nlayers = model_config['nlayers']
    nhead = model_config['nhead']  # number of heads in nn.MultiheadAttention
    dropout = model_config['dropout']  # dropout probability
    activation = model_config['activation']  # activation function

    if dataset_config['tokenizer'] == 'char':
        if dataset_config['dataset'] == 'wiki2':
            ds = wiki2(char=True)
        else:
            ds = wiki103(char=True)
    else:
        tokenizer = get_tokenizer(dataset_config['tokenizer'])
        if dataset_config['dataset'] == 'wiki2':
            ds = wiki2(tokenizer)
        elif dataset_config['dataset'] == 'wiki103':
            ds = wiki103(tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #------------------loading_dataset-----------------#
    (train_data, val_data, test_data) = ds.get_all_data()
    train_data = batchify(train_data, train_batch_size,
                          device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)
    ntokens = ds.get_vocab_len()  # size of source vocabulary
    print('vocab: {} tokens'.format(ntokens))

    if config['model'] == 'U-transformer':
        model = UTransformer(ntokens=ntokens, d_model=d_model, nhead=nhead,
                             dim_feedforward=dim_feedforward,
                             nlayers=nlayers, drop_rate=dropout, activation=activation,use_aux=use_aux\
                                 , weight=weight_aux).to(device)
    elif config['model'] == 'vanilla-transformer':
        model = VanillaTransformer(ntokens=ntokens, d_model=d_model, nhead=nhead,
                                   dim_feedforward=dim_feedforward,
                                   nlayers=nlayers, drop_rate=dropout, activation=activation).to(device)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print('-' * 89)
    print(
        f"## Training model with {pytorch_total_params/1000000:0.2F}M trainable parameters. ##")
    print('-' * 89)

    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_data)//bptt+1
    total_steps = epochs*(steps_per_epoch)
    opt_args = {
         'lr':0,
        'betas':(0.9, 0.98), 'eps':1e-9, 'weight_decay':1e-5
    }
    Noam_args ={
        'model_size':d_model,
        'factor':1, 'warmup':8000
    }
    linear_args = {
        'total_steps': total_steps,
        'pct_start':0.3, 'anneal_strategy':'linear',
        'three_phase':True, 'max_lr':1e-3
    }



    opt = torch.optim.RAdam(model.parameters(),\
         **opt_args)
    if training_config['opt'] == 'Noam':
        schedular_args = Noam_args
        optimizer = NoamOpt(**schedular_args, optimizer=opt)
    elif training_config['opt'] == 'linear':
        schedular_args = linear_args
        optimizer = linearcycleWarmup(**schedular_args, optimizer=opt)

    log_dir = time.strftime('logging/{}_%Y_%m_%d-%H_%M_%S'.format(config['model']))
    logging = training_config['logging']

    

    model, train_loss, train_ppl, val_loss, val_ppl = trainLoop(model, epochs, train_data, val_data, optimizer,
              criterion, device, bptt, clip_grad_norm, ntokens,  save_model=training_config['save_model']\
                  ,adaptive_dropout = training_config['adaptive_dropout'], logging=logging, log_dir=log_dir)

    test_loss, test_ppl = test(model, criterion, test_data, ntokens, bptt, device)

    config['opt_args'] = str(opt_args)
    config['schedular_args'] = str(schedular_args)
    config['log_dir'] = log_dir if logging else ""
    config['results'] = 'test_loss {:.3f}, val_loss {:.3f}, train_loss {:.3f}\
        test_ppl {:.3f}, val_ppl {:.3f}, train_ppl {:.3f}'.format(test_loss, val_loss, train_loss,\
            test_ppl, val_ppl, train_ppl)
    log_data('logging/log.csv', config)


    # opt = torch.optim.RAdam(model.parameters(),\
    #      lr=0, betas=(0.9, 0.98), eps=1e-9)
    #optimizer = torch.optim.RAdam(model.parameters(), lr=1.6e-6, weight_decay=1e-3)
    # optimizer = CosineAnnealingWarmupRestarts(opt,\
    #     first_cycle_steps=3*steps_per_epoch, cycle_mult=1.0, max_lr=0.001, min_lr=1e-6, warmup_steps=2*steps_per_epoch, gamma=0.5)
    #print('total_steps: {}'.format(len(train_data)//bptt))
    # optimizer = linearcycleWarmup(optimizer=opt, total_steps=5*len(train_data)//bptt,\
    #      pct_start=0.8, anneal_strategy='linear', three_phase=True,\
    #           max_lr=1e-3, steps_per_epoch=len(train_data)//bptt)
    #optimizer = linearcycleWarmup(optimizer=opt, **linear_args)

  
    


if __name__ == '__main__':
    main()
