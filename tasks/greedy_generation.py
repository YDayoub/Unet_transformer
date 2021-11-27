from torch.autograd import Variable
import torch
import torch.nn.functional as F
def greedy_decode(model, src,  max_len, generate_square_subsequent_mask, vocab,device):
    model.eval()
    for _ in range(max_len-1):
        src_mask = generate_square_subsequent_mask(src.shape[0]).to(device)
        target = torch.cat([torch.zeros(1, 1).type_as(src.data),src[:-1]], dim=0)
        output = model(src, target,src_mask, src_mask)
        prob = F.softmax(output, dim=-1)       
        preds = torch.argmax(prob,dim=-1)
        next_word = preds.data[-1,0]

        src = torch.cat([src, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    src = src.cpu().numpy()

    return ' '.join(vocab.lookup_tokens(src))

