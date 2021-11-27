import torch
from torch import nn, Tensor
import torch.nn.functional as F
from optimizers.NoamOptimizer import NoamOpt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='path to the config file', required=True)
    try:
        args = parser.parse_args()
        config_path = args.config
        config = load_config(config_path)
    except Exception as e:
        print(e)
        exit(0)
    model_config = config['model_config']
    training_config = config['training']
    eval_config = config['eval']
    dataset_config = config['dataset_config']
    #---------------loadconfig--------------------#
    train_batch_size = training_config['batch_size']
    eval_batch_size = eval_config['batch_size']
    epochs = training_config['n_epochs']
    bptt = training_config['bptt']
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
        pass
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
    if config['model'] == 'U-transformer':
        model = UTransformer(ntokens=ntokens, d_model=d_model, nhead=nhead,
                             dim_feedforward=dim_feedforward,
                             nlayers=nlayers, dropout=dropout, activation=activation).to(device)
    elif config['model'] == 'vanilla-transformer':
        model = VanillaTransformer(ntokens=ntokens, d_model=d_model, nhead=nhead,
                                   dim_feedforward=dim_feedforward,
                                   nlayers=nlayers, dropout=dropout, activation=activation).to(device)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print('-' * 89)
    print(
        f"## Training model with {pytorch_total_params/1000000:0.2F}M trainable parameters.")
    print('-' * 89)

    criterion = nn.CrossEntropyLoss()
    optimizer = NoamOpt(d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    trainLoop(model, epochs, train_data, val_data, optimizer,
              criterion, device, bptt, clip_grad_norm, ntokens,  save_model=True)

    test(model, criterion, test_data, ntokens, bptt, device)


if __name__ == '__main__':
    main()
