from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import os
import subprocess
from .char_tokenizer import CharTokenizer


class DatasetBase:
    def __init__(self, dataloader=None, tokenizer=None, char=False, data=None):
        self.char = char
        if char:
            data_path = '/data/wikitext-2-raw-v1/'
            download_data('wikitext-2-raw-v1')
            self.tokenizer = CharTokenizer(data_path)
            self.vocab = self.tokenizer.vocab
        else:
            self.tokenizer = tokenizer
            self.dataloader = dataloader
            self.vocab = self.build_vocab()

    def get_train_iter(self):
        return self.dataloader(split='train')

    def get_all_iters(self):
        if self.char:
            return self.tokenizer.get_all_iter()
        return self.dataloader()

    def get_all_data(self):
        iters = self.get_all_iters()
        datas = []
        for item in iters:
            datas.append(self.data_process(item))
        return datas

    def build_vocab(self):
        vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.get_train_iter()), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def data_process(self, raw_iter):
        if self.char:
            data = [torch.tensor(self.tokenizer(item), dtype=torch.long)
                    for item in raw_iter]
        else:
            data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long)
                    for item in raw_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def get_vocab_len(self):
        return len(self.vocab)


def download_data(ds='wikitext-2-raw-v1'):
    if not os.path.isdir("data/{}/".format(ds)):
        print("Downloading data...")
        subprocess.run(
            "wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/{}.zip -P data".format(ds).split())
        print("Unzipping data...")
        subprocess.run(["unzip", "data/{}.zip".format(ds), "-d", "data/"])
        subprocess.run(['rm', 'data/{}.zip'.format(ds)])
        print("Done...")
    else:
        print("Found data...")
