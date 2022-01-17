import torch
from utils.util import generate_square_subsequent_mask, get_batch


def evaluate(model, criterion, eval_data, ntokens, bptt, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    prev_h = None
    gru_h = None
    for i in range(0, eval_data.size(0) - 1, bptt):
        data, targets = get_batch(eval_data, i, bptt)
        batch_size = data.size(0)
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        if prev_h is None and model.use_gru:
            shape = data.shape
            prev_h = torch.zeros(
                (1, shape[1], model.d_model), device='cuda')
            gru_h = torch.zeros(
                (1, shape[1], model.d_model), device='cuda')
        outputs = model(data, src_mask, prev_h,gru_h=gru_h)
        output = outputs[0]
        if model.use_gru:
            prev_h = outputs[-1][0]
            gru_h = outputs[-1][1]
        total_loss += batch_size * (criterion(output.view(-1, ntokens), targets).item())
    return total_loss / (len(eval_data) - 1)
