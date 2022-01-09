from torchtext.datasets import WikiText103
from datasets import DatasetBase
 
class wiki103(DatasetBase):
    def __init__(self, tokenizer=None, char=False):
        data_loader = WikiText103
        super().__init__(data_loader, tokenizer, char)
    