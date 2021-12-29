import torch
from torch import nn
from copy import deepcopy
class ema(nn.Module):
    ''''
    This is a pytorch implementation of exponential moving average
    '''
    def __init__(self, model, decay_rate, zero_unbias=False, **kwargs):
        ''''
        model: basic_model, which ema is applied
        decay_rate: decay coefficient
        zero_unbias: use zero_unbais

        
        '''
        super().__init__(**kwargs)
        self.ema_model = deepcopy(model)
        self.decay_rate = decay_rate
        self.zero_unbias = zero_unbias
        if zero_unbias:
            self.updates = 15 # zero_unbiased

    @torch.no_grad()
    def forward(self, model):
        
        for ema_parm, model_parm in zip(self.ema_model.state_dict().values(),\
             model.state_dict().values()):
             ema_parm.sub_((1 - self.decay_rate) * (ema_parm - model_parm))
             if self.zero_unbias:
                self.updates += 1
                if self.updates<100:
                    ema_parm.div_(1-self.decay_rate**self.updates)
    @torch.no_grad()
    def set_model_to_ema(self, model):
      model.load_state_dict(self.ema_model.state_dict())
    def get_ema(self):
        return deepcopy(self.ema_model)

if __name__== '__main__':
    model = nn.Sequential(nn.Linear(10,5, bias=False),\
         nn.Tanh(), nn.Linear(5, 1, bias=False))
    def initialize_to_one(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.ones_(m.weight)
    model.apply(initialize_to_one)

    ema_model = ema(model, decay_rate = 0.9)
    for i in range(10):
        ema_model(model)
    ema_model.set_model_to_ema(model)
    for layer in model:
        if isinstance(layer, nn.Linear):
            print('layer: {}\n{}'.format(layer, layer.weight))

