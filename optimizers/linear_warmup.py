from .basic_optimizer import BasicOpt


class linearcycleWarmup(BasicOpt):
    def __init__(self, optimizer, schedular, *args, **kwargs):
        super().__init__(optimizer=optimizer, schedular=schedular)
        self.use_scheduler = True

       
    def step(self):
        lr_s = [p['lr'] for p in self.optimizer.param_groups]
        for p in self.optimizer.param_groups:
            p['lr']  = p['lr']*self._scalar             
        self.optimizer.step()
        for idx, p in enumerate(self.optimizer.param_groups):
            p['lr'] = lr_s[idx]
        try:
          if self.use_scheduler:
            self.schedular.step()
        except Exception as e:
          self.use_scheduler = False
          for idx, p in enumerate(self.optimizer.param_groups):
            p['lr'] = 0.00000088



    def schedule_step(self, *args):
        pass

