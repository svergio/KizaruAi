import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        att = (query @ key.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ value).transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, activation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            activation,
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class VaswaniBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ffn = FeedForward(d_model, nn.ReLU(), dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, nn.GELU(), dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _CharLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, block_size, dropout, block_cls):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [block_cls(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, time]")
        token_ids = token_ids[:, -self.block_size :]
        batch, steps = token_ids.shape
        positions = torch.arange(steps, device=token_ids.device)
        x = self.drop(self.tok(token_ids) + self.pos(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class SimpleTransformer(_CharLM):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128, dropout=0.1):
        super().__init__(vocab_size, d_model, n_layers, n_heads, block_size, dropout, VaswaniBlock)


class MiniGPT(_CharLM):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128, dropout=0.1):
        super().__init__(vocab_size, d_model, n_layers, n_heads, block_size, dropout, GPTBlock)
