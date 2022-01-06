import os
import torch
from transformers import BertTokenizer


class Corpus_subword(object):
    def __init__(self, path):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            ids = []
            for line in f:
                tokens = self.tokenizer.encode(line)
                ids = ids + tokens
        ids = torch.LongTensor(ids)
        return ids

    def get_vocab_len(self):
        return self.tokenizer.vocab_size

    def get_all_data(self):
        return self.train, self.valid, self.test
