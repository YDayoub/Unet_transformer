class CharTokenizer:
    def __init__(self,data_path):
        self.template = '{}/wiki.{{}}.raw'.format(data_path)
        self.unknown_token = '<unk>'
        train_data = self.get_data(self.template, 'train')
        self.vocab = self.build_vocab(train_data)
       # del train_data
    def build_vocab(self, data):
        vocab = dict()
        for line in data:
            for item in line:
                vocab[item] = vocab.get(item,len(vocab))
        vocab[self.unknown_token] = len(vocab)
        return vocab
    def __call__(self, line) :
        return list(self.vocab.get(item,\
            self.vocab[self.unknown_token]) for item in line)
    def get_data(self,template, split):
        file_path = template.format(split) 
        with open(file_path,'rb') as f:
            lines = f.readlines()
        return lines
    def get_all_iter(self):
        data = [self.get_data(self.template, 'train'),\
                self.get_data(self.template, 'valid'),\
                self.get_data(self.template,'test')]
        return data

