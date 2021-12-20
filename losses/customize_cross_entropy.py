import torch
from torch.nn import functional as F
class custom_ce_loss:
    def __init__(self, num_classes, power=1):
        self.power = power
        self.num_classes = num_classes
    def __call__(self, logits, targets):
        losses = F.cross_entropy(logits, targets, reduction='none')
        
        return torch.mean(losses**2)
