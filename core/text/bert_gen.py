import torch
from torch import nn

MASK_TOKEN = "[MASK]"


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        att = (query @ key.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ value).transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(out)


class BertBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = BidirectionalAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class BertMLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        block_size=256,
        dropout=0.1,
        mask_id=None,
        embedding_size=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.embedding_size = int(embedding_size or d_model)
        self.mask_id = vocab_size - 1 if mask_id is None else int(mask_id)
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, self.embedding_size)
        self.emb_proj = (
            nn.Identity()
            if self.embedding_size == d_model
            else nn.Linear(self.embedding_size, d_model, bias=False)
        )
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [BertBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, time]")
        token_ids = token_ids[:, : self.block_size]
        batch, steps = token_ids.shape
        positions = torch.arange(steps, device=token_ids.device)
        x = self.drop(self.emb_proj(self.tok(token_ids)) + self.pos(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def apply_mlm_mask(tokens, mask_id, vocab_size, mlm_prob=0.15):
    labels = tokens.clone()
    probability = torch.full(tokens.shape, mlm_prob, device=tokens.device)
    chosen = torch.bernoulli(probability).bool()
    if not chosen.any():
        chosen[0, 0] = True
    labels = labels.masked_fill(~chosen, -100)
    inputs = tokens.clone()
    rand = torch.rand(tokens.shape, device=tokens.device)
    inputs = torch.where(chosen & (rand < 0.8), torch.full_like(inputs, mask_id), inputs)
    random_token = torch.randint(0, vocab_size - 1, tokens.shape, device=tokens.device)
    inputs = torch.where(chosen & (rand >= 0.8) & (rand < 0.9), random_token, inputs)
    return inputs, labels


def _sample_id(logits, temp, mask_id):
    logits = logits.clone()
    logits[mask_id] = float("-inf")
    if temp <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temp, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_mlm(
    model,
    char_to_idx,
    idx_to_char,
    start_text="Я",
    seq_len=None,
    n_iters=32,
    temp=1.0,
    device=None,
    max_lines=6,
    level="char",
):
    from core.text.tokenize import TOKEN_NL, detokenize, encode_words

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    seq_len = int(seq_len or min(128, model.block_size))
    seq_len = min(seq_len, model.block_size)
    if level == "word":
        seed = encode_words(start_text or "", char_to_idx)
    else:
        seed = [char_to_idx[ch] for ch in (start_text or "") if ch in char_to_idx]
    frozen = min(len(seed), seq_len)
    ids = seed[:frozen] + [model.mask_id] * (seq_len - frozen)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    free = list(range(frozen, seq_len))
    for pos in free:
        tokens[0, pos] = model.mask_id
        logits = model(tokens)[0, pos]
        tokens[0, pos] = _sample_id(logits, temp, model.mask_id)
    for _ in range(max(0, int(n_iters))):
        from core.bus.cancel_flag import check_cancelled

        check_cancelled()
        if not free:
            break
        pos = free[int(torch.randint(0, len(free), (1,)).item())]
        tokens[0, pos] = model.mask_id
        logits = model(tokens)[0, pos]
        tokens[0, pos] = _sample_id(logits, temp, model.mask_id)
    if level == "word":
        out_tokens = []
        lines = 0
        for idx in tokens[0].tolist():
            if idx == model.mask_id:
                continue
            tok = idx_to_char.get(int(idx), "")
            if not tok:
                continue
            out_tokens.append(tok)
            if tok == TOKEN_NL:
                lines += 1
                if max_lines is not None and lines >= max_lines:
                    break
        return detokenize(out_tokens).rstrip()
    chars = []
    lines = 0
    for idx in tokens[0].tolist():
        if idx == model.mask_id:
            continue
        ch = idx_to_char.get(int(idx), "")
        if ch == "\n" and chars and chars[-1] == "\n":
            continue
        chars.append(ch)
        if ch == "\n":
            lines += 1
            if max_lines is not None and lines >= max_lines:
                break
    return "".join(chars).rstrip()
