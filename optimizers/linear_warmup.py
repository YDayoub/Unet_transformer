from torch.optim.lr_scheduler import OneCycleLR
class linearcycleWarmup:
    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.schedular = OneCycleLR(self.optimizer, *args, **kwargs)
        self.counter = 0
        self.epochs = 1
        self.parameters = kwargs
        self.max_lr = self.parameters['max_lr']
        self.threshold = int(kwargs['total_steps']*kwargs['pct_start'])
        #self.min_lr = 0.00012


    @property
    def lr(self):
      return self.schedular.get_last_lr()[0]
        
    def step(self):
        
        self.optimizer.step()
        # self.counter+=1
        # if self.counter>self.threshold:
        #     if self.lr<=self.min_lr:
        #         return
        self.schedular.step()
        # self.counter+=1
        # if self.counter >= self.parameters['total_steps']:
        #     self.counter = 0
        #     self.epochs += self.parameters['total_steps']//self.parameters['steps_per_epoch']
        #     self.parameters['max_lr'] = self.max_lr/self.epochs
        #     self.schedular =  OneCycleLR(self.optimizer, **self.parameters)


    def zero_grad(self):
        self.optimizer.zero_grad()

