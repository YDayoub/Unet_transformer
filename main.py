import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from models import create_model
from optimizers import create_optimizer
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
import os
import warnings
from losses import custom_ce_loss
warnings.filterwarnings('ignore')
from datasets.Data import Corpus
from tasks.pointer import evaluate_pointer


def get_activation(act_func:str='relu'):
    if act_func=='relu':
        return nn.functional.relu_
    elif act_func == 'gelu':
        return nn.functional.gelu
    elif act_func == 'elu':
        return nn.functional.elu
    elif act_func == 'leaky_relu':
        return nn.functional.leaky_relu
    else:
        raise Exception('Please specify valid activaiton function')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='path to the config file', required=True)
    parser.add_argument(
        '-f', '--flag', help='boolean to continue training', default=False)
    parser.add_argument(
        '-p', '--path', help='path to the checkpoint file', default='./checkpoints'
    )
    try:
        args = parser.parse_args()
        config_path = args.config
        continue_training = args.flag
        checkpoint_path = args.path
        if continue_training:
            if os.path.isfile(checkpoint_path):
                model, optimizer,  start_epoch, config = load_model(checkpoint_path)
            else:
                raise Exception('checkpoint file not found, please check the path and rerun')
        else:
            config = load_config(config_path)
        

    except Exception as e:
        print('Error', e)
        exit(0)





    #--------------- Reproducibility -------------#
    #set_seed(1111)
    config = {**config, 'seed': 1111}
    print(config)
    #---------------                  -------------#
    model_config = config['model_config']
    training_config = config['training']
    eval_config = config['eval']
    dataset_config = config['dataset_config']
    #---------------loadconfig--------------------#
    train_batch_size = training_config['batch_size']
    eval_batch_size = eval_config['batch_size']
    epochs = training_config['n_epochs']
    start_epoch = 1 if not 'start_epoch' in vars() else start_epoch
    bptt = training_config['bptt']
    use_aux = training_config['use_aux']
    weight_aux = training_config['weight_aux']
    use_var_len = training_config['use_var_len']
    weight_decay = float(training_config['weight_decay'])
    clip_grad_norm = training_config['clip_grad_norm']
    d_model = model_config['d_model']  # embedding dimension
    dim_feedforward = model_config['dim_feedforward'] 
    nlayers = model_config['nlayers']
    nhead = model_config['nhead']  # number of heads in nn.MultiheadAttention
    dropout = model_config['dropout']  # dropout probability
    emb_dropout = model_config['emb_dropout']
    out_dropout = model_config['out_dropout']
    in_dropout = model_config['in_dropout']
    activation = get_activation(model_config['activation'])  # activation function
    
    tying = model_config['tying']
    alpha = float(training_config['alpha'])
    beta = float(training_config['beta'])
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
    print('Training On Device: {}'.format(device))
    #------------------loading_dataset-----------------#
    (train_data, val_data, test_data) = ds.get_all_data()
    # ds = Corpus('/home/admin/datasets/wikitext-2')
    # (train_data, val_data, test_data) = ds.train, ds.valid, ds.test
    train_data = batchify(train_data, train_batch_size,
                          device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)
    ntokens = ds.get_vocab_len()  # size of source vocabulary
    print('vocab: {} tokens'.format(ntokens))
    model_args ={'ntokens': ntokens, 'd_model':d_model, 'nhead':nhead,
                 'dim_feedforward': dim_feedforward, 'nlayers': nlayers,
                 'drop_rate': dropout, 'activation': activation, 'use_aux':use_aux\
                , 'weight': weight_aux, 'tying': tying, 'emb_dropout':emb_dropout,
                
                'in_dropout': in_dropout,  'out_dropout': out_dropout,
                
                }
    model = create_model(config['model'],**model_args).to(device) if not 'model' in vars() else model

    
    #evaluate_pointer(model, val_data, ntokens, device)
    
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print('-' * 89)
    print(
        '#'*12+f" Training model with {pytorch_total_params/1000000:0.2F}M trainable parameters for {epochs:3d} epochs "+'#'*12)
    print('-' * 89)

    criterion = nn.CrossEntropyLoss()
    custom_loss = custom_ce_loss(num_classes=ntokens, power=2)
    steps_per_epoch = len(train_data)//bptt+1
    total_steps = epochs*(steps_per_epoch)
    opt_args = {
         'lr':0,
        'betas':(0.9, 0.98), 'eps':1e-9, 'weight_decay': weight_decay
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

    Reduce_on_Plateua_args = {
        'mode':'min',
         'factor':0.1, 'patience': 3
    }




    opt = torch.optim.RAdam(model.parameters(),\
         **opt_args)
    

    if training_config['opt'] == 'NoamOptimizer':
        schedular_args = Noam_args
        args = {**schedular_args, 'optimizer':opt}
    elif training_config['opt'] == 'linear':
        schedular_args = linear_args
        schedular = OneCycleLR(optimizer=opt, **schedular_args)
        args = {'schedular': schedular, 'optimizer':opt}
    elif training_config['opt'] == 'sgd_platue':
        opt = torch.optim.SGD(model.parameters(), lr=5, weight_decay = weight_decay)
        schedular = ReduceLROnPlateau(optimizer=opt, **Reduce_on_Plateua_args)
        args = {'optimizer':opt, 'schedular': schedular} 
        opt.param_groups[0]['lr'] = 5
    elif training_config['opt'] == 'sgd_lr':
        opt = torch.optim.SGD(model.parameters(), lr = 5)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
        args = {'optimizer':opt, 'schedular': scheduler} 
    optimizer = create_optimizer(training_config['opt'],**args) if not 'optimizer' in vars() else optimizer

    log_dir = time.strftime('logging/{}_%Y_%m_%d-%H_%M_%S'.format(config['model']))
    logging = training_config['logging']
 
    if continue_training:
        print('Continuing training from {} to {} for {} model'.format(\
            start_epoch, epochs, model.model_type))


    

    
    model, train_loss, train_ppl, val_loss, val_ppl = trainLoop(model, config, start_epoch, epochs, train_data, val_data, optimizer,
              criterion, device, bptt, clip_grad_norm, ntokens,alpha=alpha, beta=beta,  save_model_flag=training_config['save_model']\
                  ,adaptive_dropout = training_config['adaptive_dropout'], logging=logging, log_dir=log_dir,\
                      use_var_len=use_var_len, custom_loss = None)


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
    try:
        main()
    except KeyboardInterrupt:
        import sys
        print('keyboard itnerrupt has been called')
        
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
