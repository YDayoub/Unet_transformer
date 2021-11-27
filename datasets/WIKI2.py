from torchtext.datasets import WikiText2
from datasets import DatasetBase

class wiki2(DatasetBase):
    def __init__(self,*args,**kwargs):
        data_loader = WikiText2
        super().__init__(data_loader,*args, **kwargs)

    