from torchtext.datasets import WikiText2
from datasets import DatasetBase

class wiki2(DatasetBase):
    def __init__(self, tokenizer=None, char=False):
        if char:
            data_loader = None
        else:
            data_loader = WikiText2
        super().__init__(data_loader, tokenizer, char)

    