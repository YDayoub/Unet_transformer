import torch
from torch.optim import ASGD

class myASGD:   
    def __init__(self, optimizer, *args,**kwargs):
        self.optim = optimizer

    def step(self):
        self.optim.step()
    def zero_grad(self):
        self.optim.zero_grad()
    @property
    def lr(self):
        p = self.optimizer.param_groups[0]
        self._lr = p['lr']
        return self._lr

