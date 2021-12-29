import torch
from utils.util import generate_square_subsequent_mask, get_batch


def evaluate(model, criterion, eval_data, ntokens, bptt, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    prev_h = None
    for i in range(0, eval_data.size(0) - 1, bptt):
        data, targets = get_batch(eval_data, i, bptt)
        batch_size = data.size(0)
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        if prev_h is None and model.save_state:
            shape = data.shape
            prev_h = torch.randn(
                (shape[0], shape[1], model.d_model), device='cuda')
        elif prev_h is not None and prev_h.shape[0] != batch_size:
            prev_h = prev_h[-batch_size:,:,:]
        outputs = model(data, src_mask, prev_h)
        output = outputs[0]
        if model.save_state:
            prev_h = outputs[-1]
        total_loss += batch_size * (criterion(output.view(-1, ntokens), targets).item())
    return total_loss / (len(eval_data) - 1)
