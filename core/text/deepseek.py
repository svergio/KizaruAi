import torch
from torch import nn
from torch.nn import functional as F


def _lora_rank(d_model, override=None):
    if override is not None:
        return int(override)
    return max(16, d_model // 4)


def _swiglu_hidden(d_model, mult=8 / 3):
    hidden = int(mult * d_model)
    return max(d_model, hidden - (hidden % 64) if hidden >= 64 else hidden)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden = _swiglu_hidden(d_model)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MultiHeadLatentAttention(nn.Module):
    """Compressed Q/KV attention (DeepSeek MLA sketch, no MoE / no RoPE ranks)."""

    def __init__(
        self,
        d_model,
        n_heads,
        block_size,
        kv_lora_rank=None,
        q_lora_rank=None,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kv_lora_rank = _lora_rank(d_model, kv_lora_rank)
        self.q_lora_rank = _lora_rank(d_model, q_lora_rank)
        self.wq_a = nn.Linear(d_model, self.q_lora_rank, bias=False)
        self.wq_b = nn.Linear(self.q_lora_rank, n_heads * self.d_head, bias=False)
        self.wkv_a = nn.Linear(d_model, self.kv_lora_rank, bias=False)
        self.wkv_b = nn.Linear(self.kv_lora_rank, 2 * n_heads * self.d_head, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch, steps, width = x.shape
        query = self.wq_b(self.wq_a(x))
        query = query.view(batch, steps, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.wkv_b(self.wkv_a(x))
        kv = kv.view(batch, steps, 2, self.n_heads, self.d_head)
        key = kv[:, :, 0].transpose(1, 2)
        value = kv[:, :, 1].transpose(1, 2)
        att = (query @ key.transpose(-2, -1)) * (self.d_head**-0.5)
        att = att.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ value).transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(out)


class DeepSeekBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        block_size,
        kv_lora_rank=None,
        q_lora_rank=None,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model, n_heads, block_size, kv_lora_rank, q_lora_rank, dropout
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniDeepSeek(nn.Module):
    """Small DeepSeek-style char LM: MLA + RMSNorm + SwiGLU, ~15-50M with train defaults."""

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        block_size=128,
        dropout=0.1,
        kv_lora_rank=None,
        q_lora_rank=None,
        embedding_size=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.embedding_size = int(embedding_size or d_model)
        self.kv_lora_rank = _lora_rank(d_model, kv_lora_rank)
        self.q_lora_rank = _lora_rank(d_model, q_lora_rank)
        self.tok = nn.Embedding(vocab_size, self.embedding_size)
        self.emb_proj = (
            nn.Identity()
            if self.embedding_size == d_model
            else nn.Linear(self.embedding_size, d_model, bias=False)
        )
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DeepSeekBlock(
                    d_model,
                    n_heads,
                    block_size,
                    self.kv_lora_rank,
                    self.q_lora_rank,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, time]")
        token_ids = token_ids[:, -self.block_size :]
        batch, steps = token_ids.shape
        positions = torch.arange(steps, device=token_ids.device)
        x = self.drop(self.emb_proj(self.tok(token_ids)) + self.pos(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))


try:
    from transformers import StoppingCriteria as _HFStoppingCriteria
except Exception:  # pragma: no cover - optional dependency shape
    class _HFStoppingCriteria:  # type: ignore[no-redef]
        def __call__(self, input_ids, scores, **kwargs):
            return False


class CancelFlagStoppingCriteria(_HFStoppingCriteria):
    """HF StoppingCriteria that reads core.bus.cancel_flag (cancel / MAX_RUNTIME)."""

    def __call__(self, input_ids, scores, **kwargs):
        from core.bus.cancel_flag import get_cancel_event

        event = get_cancel_event()
        return event is not None and event.is_set()


def generate(
    model,
    char_to_idx,
    idx_to_char,
    start_text="Я",
    prediction_len=48,
    temp=0.8,
    device=None,
    max_lines=None,
    level="char",
    stopping_criteria=None,
):
    """Autoregressive generate with HF-style StoppingCriteria (cancel_flag)."""
    from core.bus.cancel_flag import GenerationCancelled
    from core.text.sample import (
        _completed_lyric_lines,
        _completed_lyric_lines_tokens,
        _encode_start,
        _sample_index,
    )

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    criteria = list(stopping_criteria or [])
    criteria.append(CancelFlagStoppingCriteria())

    def _stop(ids):
        tensor_ids = torch.tensor([ids], dtype=torch.long, device=device)
        return any(bool(crit(tensor_ids, None)) for crit in criteria)

    if level == "word":
        from core.text.tokenize import TOKEN_UNK, detokenize, encode_words

        ids = encode_words(start_text, char_to_idx)
        start_lines = _completed_lyric_lines_tokens(ids, idx_to_char)
        block = [char_to_idx[TOKEN_UNK]] if TOKEN_UNK in char_to_idx else None
        block_size = model.block_size
        with torch.no_grad():
            for _ in range(prediction_len):
                if _stop(ids):
                    raise GenerationCancelled()
                ctx = torch.LongTensor(ids[-block_size:]).view(1, -1).to(device)
                logits = model(ctx)[0, -1]
                probs = F.softmax(logits / max(temp, 1e-6), dim=-1).cpu().numpy()
                top_index = _sample_index(probs, len(char_to_idx), block_ids=block)
                ids.append(top_index)
                if (
                    max_lines is not None
                    and _completed_lyric_lines_tokens(ids, idx_to_char) - start_lines
                    >= max_lines
                ):
                    break
        tokens = [
            idx_to_char[int(idx)]
            for idx in ids
            if idx_to_char.get(int(idx)) != TOKEN_UNK
        ]
        return detokenize(tokens).rstrip()

    ids = _encode_start(start_text, char_to_idx)
    predicted_text = "".join(idx_to_char[int(idx)] for idx in ids)
    start_lines = _completed_lyric_lines(predicted_text)
    block_size = model.block_size
    with torch.no_grad():
        for _ in range(prediction_len):
            if _stop(ids):
                raise GenerationCancelled()
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
