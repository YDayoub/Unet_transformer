class BasicOpt:
    def __init__(self, optimizer, schedular):
        self.optimizer = optimizer
        self.schedular = schedular
        self._scalar = 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def lr(self):
      return self.optimizer.param_groups[0]['lr']*self._scalar

    @property
    def scalar(self):
        return self._scalar

    @scalar.setter
    def scalar(self, scalar):
        self._scalar = scalar


    def schedule_step(self, val_loss):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError