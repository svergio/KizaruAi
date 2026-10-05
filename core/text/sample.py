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


def _completed_lyric_lines(text: str) -> int:
    """Non-empty lines closed by a newline; blank lines do not count."""
    if not text:
        return 0
    return sum(1 for line in text.split("\n")[:-1] if line.strip())


def _sample_index(probs, vocab_size, block_ids=None):
    probs = np.asarray(probs, dtype=np.float64).copy()
    if probs.shape[0] > vocab_size:
        probs = probs[:vocab_size]
    if block_ids:
        for idx in block_ids:
            i = int(idx)
            if 0 <= i < probs.shape[0]:
                probs[i] = 0.0
    total = float(probs.sum())
    if total <= 0:
        return int(np.random.randint(0, probs.shape[0]))
    probs = probs / total
    return int(np.random.choice(probs.shape[0], p=probs))


def _completed_lyric_lines_tokens(ids, idx_to_token) -> int:
    from core.text.tokenize import TOKEN_NL

    count = 0
    has_content = False
    for idx in ids:
        if idx_to_token.get(int(idx)) == TOKEN_NL:
            if has_content:
                count += 1
            has_content = False
        else:
            has_content = True
    return count


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
    start_lines = _completed_lyric_lines(predicted_text)

    with torch.no_grad():
        output = None
        for idx in idx_input:
            output, hidden = model.step(torch.LongTensor([idx]).to(device), hidden)
        for _ in range(prediction_len):
            # один символ из текущего hidden; следующий шаг ест уже выбранный токен
            logits = output.reshape(-1).cpu()
            probs = F.softmax(logits / max(temp, 1e-6), dim=-1).numpy()
            top_index = _sample_index(probs, len(char_to_idx))
            predicted_text += idx_to_char[top_index]
            output, hidden = model.step(torch.LongTensor([top_index]).to(device), hidden)
            if (
                max_lines is not None
                and _completed_lyric_lines(predicted_text) - start_lines >= max_lines
            ):
                break
    return predicted_text.rstrip()


def evaluate_words(
    model,
    token_to_idx,
    idx_to_token,
    start_text="Я",
    prediction_len=48,
    temp=0.8,
    device=None,
    max_lines=None,
):
    from core.text.tokenize import TOKEN_UNK, detokenize, encode_words

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device=device)
    ids = encode_words(start_text, token_to_idx)
    start_lines = _completed_lyric_lines_tokens(ids, idx_to_token)
    block = [token_to_idx[TOKEN_UNK]] if TOKEN_UNK in token_to_idx else None

    with torch.no_grad():
        output = None
        for idx in ids:
            output, hidden = model.step(torch.LongTensor([idx]).to(device), hidden)
        for _ in range(prediction_len):
            logits = output.reshape(-1).cpu()
            probs = F.softmax(logits / max(temp, 1e-6), dim=-1).numpy()
            top_index = _sample_index(probs, len(token_to_idx), block_ids=block)
            ids.append(top_index)
            output, hidden = model.step(torch.LongTensor([top_index]).to(device), hidden)
            if (
                max_lines is not None
                and _completed_lyric_lines_tokens(ids, idx_to_token) - start_lines
                >= max_lines
            ):
                break
    tokens = [
        idx_to_token[int(idx)]
        for idx in ids
        if idx_to_token.get(int(idx)) != TOKEN_UNK
    ]
    return detokenize(tokens).rstrip()


def evaluate_lm(
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
    ids = _encode_start(start_text, char_to_idx)
    predicted_text = "".join(idx_to_char[int(idx)] for idx in ids)
    start_lines = _completed_lyric_lines(predicted_text)
    block_size = model.block_size

    with torch.no_grad():
        for _ in range(prediction_len):
            ctx = torch.LongTensor(ids[-block_size:]).view(1, -1).to(device)
            logits = model(ctx)[0, -1]
            probs = F.softmax(logits / max(temp, 1e-6), dim=-1).cpu().numpy()
            top_index = _sample_index(probs, len(char_to_idx))
            ids.append(top_index)
            predicted_text += idx_to_char[top_index]
            if (
                max_lines is not None
                and _completed_lyric_lines(predicted_text) - start_lines >= max_lines
            ):
                break
    return predicted_text.rstrip()


def evaluate_lm_words(
    model,
    token_to_idx,
    idx_to_token,
    start_text="Я",
    prediction_len=48,
    temp=0.8,
    device=None,
    max_lines=None,
):
    from core.text.tokenize import TOKEN_UNK, detokenize, encode_words

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = encode_words(start_text, token_to_idx)
    start_lines = _completed_lyric_lines_tokens(ids, idx_to_token)
    block_size = model.block_size
    block = [token_to_idx[TOKEN_UNK]] if TOKEN_UNK in token_to_idx else None

    with torch.no_grad():
        for _ in range(prediction_len):
            ctx = torch.LongTensor(ids[-block_size:]).view(1, -1).to(device)
            logits = model(ctx)[0, -1]
            probs = F.softmax(logits / max(temp, 1e-6), dim=-1).cpu().numpy()
            top_index = _sample_index(probs, len(token_to_idx), block_ids=block)
            ids.append(top_index)
            if (
                max_lines is not None
                and _completed_lyric_lines_tokens(ids, idx_to_token) - start_lines
                >= max_lines
            ):
                break
    tokens = [
        idx_to_token[int(idx)]
        for idx in ids
        if idx_to_token.get(int(idx)) != TOKEN_UNK
    ]
    return detokenize(tokens).rstrip()
