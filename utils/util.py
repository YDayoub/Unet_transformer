import torch
import yaml
import os
import numpy as np
import random
import matplotlib as mpl
from matplotlib import pyplot as plt


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def batchify(data, bsz, device):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def load_config(file_path='config.yaml'):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_fig(fpath, tight_layout=True, fig_extension="png", resolution=300):

    print("Saving figure", fpath)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fpath, format=fig_extension, dpi=resolution)


def save_model(checkpoint, path):
    '''
    function to save model
    Args:
        checkpoint: a dictionary which contians:
        'configs': configs
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
    
    '''
    torch.save(checkpoint, path)


def load_model(fpath):
    '''
    path to the checkpoint file
    the checkpoint file should contains the following:
        'configs': configs
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,

    '''
    checkpoint = torch.load(fpath)
    epoch = checkpoint['epoch']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    configs = checkpoint['configs']
    return model, optimizer, epoch, configs 



