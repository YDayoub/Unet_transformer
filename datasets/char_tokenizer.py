class CharTokenizer:
    def __init__(self,data_path):
        self.template = '{}/wiki.{}.raw'.format(data_path)
        train_data = self.get_data(self.template, 'train')
        self.voacb = self.build_voacb(train_data)
        del train_data
        self.unknown_token = '<unk>'
    def build_voacb(self, data):
        vocab = dict()
        [vocab.get(item,len(vocab)) \
            for line in data for item in line]
        vocab[self.unknown_token] = len(vocab)
        return vocab
    def __call__(self, line) :
        return list(self.voacb.get(item,\
            self.voacb[self.unknown_token]) for item in line)
    def get_data(self,template, split):
        file_path = template.format(split) 
        with open(file_path,'rb') as f:
            lines = f.readlines()
        return lines
    def get_all_iter(self):
        data = [self.get_data(self.template, 'train'),\
                self.get_data(self.template, 'valid'),\
                self.get_data(self.tempalate,'test')]
        return data

