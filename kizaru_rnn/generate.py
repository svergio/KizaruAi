import numpy as np
import torch
from torch.nn import functional as F


def _encode_start(start_text, char_to_idx):
    ids = [char_to_idx[char] for char in start_text if char in char_to_idx]
    if ids:
        return ids
    if " " in char_to_idx:
        return [char_to_idx[" "]]
    return [0]


def evaluate(
    model,
    char_to_idx,
    idx_to_char,
    start_text=" ",
    prediction_len=200,
    temp=0.3,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device=device)
    idx_input = _encode_start(start_text, char_to_idx)
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = "".join(idx_to_char[idx] for idx in idx_input)

    with torch.no_grad():
        _, hidden = model(train, hidden)
        inp = train[-1].view(-1, 1, 1)
        for _ in range(prediction_len):
            output, hidden = model(inp, hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / max(temp, 1e-6), dim=-1).numpy()
            top_index = int(np.random.choice(len(char_to_idx), p=p_next))
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_text += idx_to_char[top_index]
    return predicted_text
