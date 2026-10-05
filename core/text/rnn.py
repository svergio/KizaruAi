import torch
from torch import nn


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        # x: [time, batch, 1] — внутри LSTM всё равно идёт по времени, не «вся фраза разом»
        x = self.encoder(x).squeeze(2)
        out, (ht, ct) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht, ct)

    def step(self, token, hidden):
        # Один символ: logits следующего и новый hidden. Генерация только так.
        if token.dim() == 1:
            token = token.view(1, -1, 1)
        elif token.dim() == 2:
            token = token.view(1, token.shape[0], 1)
        elif token.dim() != 3:
            raise ValueError("token must be [batch], [batch,1] or [1,batch,1]")
        return self.forward(token, hidden)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
