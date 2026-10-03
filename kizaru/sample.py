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
    max_lines=None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device=device)
    idx_input = _encode_start(start_text, char_to_idx)
    seed = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = "".join(idx_to_char[int(idx)] for idx in idx_input)
    start_lines = predicted_text.count("\n")

    with torch.no_grad():
        _, hidden = model(seed, hidden)
        current = seed[-1].view(-1, 1, 1)
        for _ in range(prediction_len):
            output, hidden = model(current, hidden)
            logits = output.cpu().data.view(-1)
            probs = F.softmax(logits / max(temp, 1e-6), dim=-1).numpy()
            top_index = int(np.random.choice(len(char_to_idx), p=probs))
            current = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_text += idx_to_char[top_index]
            if max_lines is not None and predicted_text.count("\n") - start_lines >= max_lines:
                break
    return predicted_text.rstrip()
