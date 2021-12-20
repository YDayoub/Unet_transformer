import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


from utils.util import generate_square_subsequent_mask, get_batch

def one_hot(idx, size, cuda=True):
    a = np.zeros((1, size), np.float32)
    a[0][idx] = 1
    v = Variable(torch.from_numpy(a.reshape((1,size))))
    if cuda: v = v.cuda()
    return v

def evaluate_pointer(model, data_source, ntokens, device, batch_size=10, window=100, bptt=128, labmdsasm=0.12785920428335693,theta=0.6625523432485668):
        criterion = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0
        next_word_history = None
        pointer_history = None
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            x = one_hot(targets.data[0], ntokens)
            print('x_shape', x.shape)
            batch_size = data.size(0)
            src_mask = generate_square_subsequent_mask(batch_size).to(device)
            if model.use_aux:
                output, _, hiddens = model(data, src_mask)
            else:
                output, hiddens = model(data, src_mask)
            rnn_out = hiddens[-1].squeeze()
            print('rnn_out shape {}'.format(rnn_out.shape))
            output_flat = output.view(-1, ntokens)
            start_idx = len(next_word_history) if next_word_history is not None else 0
            next_word_history = torch.cat([one_hot(t, ntokens) for t in targets])\
                 if next_word_history is None else\
                      torch.cat([next_word_history, torch.cat([one_hot(t, ntokens)\
                           for t in targets])])
            print('next_word_history.shape {}'.format(next_word_history.shape))
            pointer_history = Variable(rnn_out.data)\
                 if pointer_history is None\
                      else torch.cat([pointer_history, Variable(rnn_out.data)], dim=0)

            print('pointer_history.shape {}'.format(pointer_history.shape))

            loss = 0
            softmax_output_flat = torch.nn.functional.softmax(output_flat)
            for idx, vocab_loss in enumerate(softmax_output_flat):
                p = vocab_loss
                if start_idx + idx > window:
                    valid_next_word = next_word_history[start_idx + idx - window:start_idx + idx]
                    valid_pointer_history = pointer_history[start_idx + idx - window:start_idx + idx]
                    print('rnn_out[idx] shape {}'.format(rnn_out[idx].shape))
                    print('valid_pointer_history.shape {}'.format(valid_pointer_history.shape))
                    logits = torch.mv(valid_pointer_history, rnn_out[idx])
                    ptr_attn = torch.nn.functional.softmax(theta * logits).view(-1, 1)
                    ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
                    p = labmdsasm * ptr_dist + (1 - labmdsasm) * vocab_loss
                ###
                target_loss = p[targets[idx].data]
                loss += (-torch.log(target_loss)).item()
            total_loss += loss / batch_size
            ###

            next_word_history = next_word_history[-window:]
            pointer_history = pointer_history[-window:]
        return total_loss / len(data_source)