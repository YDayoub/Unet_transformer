from __future__ import absolute_import

from .linear_warmup import linearcycleWarmup
from .NoamOptimizer import NoamOpt
from .sgd_platue import SGDRPLateu
from .Sgd_lr import SGDLR


__factory = {
    'linear': linearcycleWarmup,
    'NoamOptimizer': NoamOpt,
    'sgd_platue': SGDRPLateu,
    'sgd_lr': SGDLR
}


def names():
    return sorted(__factory.keys())


def create_optimizer(name, *args, **kwargs):
    
    if name not in __factory:
        raise KeyError("Unknown optimizer:", name)
    return __factory[name](*args, **kwargs)



