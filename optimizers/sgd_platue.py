from .basic_optimizer import BasicOpt

class SGDRPLateu(BasicOpt):
    def __init__(self, optimizer, schedular, *args,  **kwargs):
        super().__init__(optimizer=optimizer, schedular=schedular)
        
    def step(self):
        lr_s = [p['lr'] for p in self.optimizer.param_groups]

        for p in self.optimizer.param_groups:
            p['lr']  = p['lr']*self._scalar
        
        self.optimizer.step()
        for idx, p in enumerate(self.optimizer.param_groups):
            p['lr'] = lr_s[idx] 

    def schedule_step(self, val_loss):
        self.schedular.step(val_loss)