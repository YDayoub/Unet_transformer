from .basic_optimizer import BasicOpt

class NoamOpt(BasicOpt):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, schedular=None, factor=1, warmup=8000, model_size= 512):
        super().__init__(optimizer, schedular)
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._lr = 0
        self._step = 0
    @property
    def lr(self):
      return self._lr
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()*self._scalar
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._lr = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


    def schedule_step(self, *args):
        pass
