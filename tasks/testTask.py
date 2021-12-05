from .evalTask import evaluate
import math
def test(model, criterion,  eval_data, ntoken_tgt, bptt, device):
    test_loss = evaluate(model, criterion,  eval_data, ntoken_tgt, bptt, device)
    test_ppl = math.exp(test_loss)
    print('-' * 89)
    print(f'| test loss {test_loss:5.5f} | test ppl {test_ppl:8.2f}')
    print('-' * 89)
    return test_loss, test_ppl
    
