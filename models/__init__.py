from __future__ import absolute_import

from .UNet_transformer import UTransformer
from .vanilla_transformer import VanillaTransformer


__factory = {
    'vanilla-transformer': VanillaTransformer,
    'U-transformer': UTransformer,
}


def names():
    return sorted(__factory.keys())

def create_model(name, *args, **kwargs):
    
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)



