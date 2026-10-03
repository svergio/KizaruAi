import random
import re
from pathlib import Path

import torch
from torch import nn

# Phrases from notebooks/определенее настроение).ipynb
NOTEBOOK_TRAIN = {
    "хороший": True,
    "круто": True,
    "весело": True,
    "ахуенно": True,
    "ултра хорош": True,
    "супер мега хорош": True,
    "неплохо": True,
    "не грустно": True,
    "очень хорошо": True,
    "счаслив": True,
    "я хорош": True,
    "я охуителен": True,
    "мне хорошо": True,
    "мне не плохо": True,
    "Это очень хорошо": True,
    "хуево": False,
    "насрал": False,
    "плохо": False,
    "не хорошо": False,
    "кринжанул": False,
    "ультра насрал": False,
    "не радует": False,
    "радует": True,
    "грустно": False,
}

NOTEBOOK_TEST = {
    "это счастье": True,
    "я хороший": True,
    "я не хороший": False,
    "мне не грустно": True,
    "я очень хорош": True,
    "мне очень грустно": False,
    "это плохо, а не хорошо": False,
}

EXTRA_TRAIN = {
    "счастье": True,
    "кайф": True,
    "люблю": True,
    "огонь": True,
    "класс": True,
    "радость": True,
    "спасибо": True,
    "обожаю": True,
    "топ": True,
    "легенда": True,
    "красиво": True,
    "приятно": True,
    "не плохо": True,
    "не хуево": True,
    "не печально": True,
    "очень круто": True,
    "я счастлив": True,
    "мне весело": True,
    "это огонь": True,
    "все хорошо": True,
    "кайфово": True,
    "а круто": True,
    "печально": False,
    "бесит": False,
    "ненавижу": False,
    "отстой": False,
    "ужас": False,
    "больно": False,
    "злой": False,
    "тоска": False,
    "мерзко": False,
    "кринж": False,
    "отвратительно": False,
    "не хороший": False,
    "совсем не хороший": False,
    "не ахуенно": False,
    "не огонь": False,
    "не кайф": False,
    "я не огонь": False,
    "не круто": False,
    "не весело": False,
    "вовсе не грустно": True,
    "вовсе не плохо": True,
    "мне плохо": False,
    "очень плохо": False,
    "это ужас": False,
    "все плохо": False,
    "а хуево": False,
    "я устал": False,
    "достало": False,
}

UNK = "<unk>"
POSITIVE = "позитив"
NEGATIVE = "негатив"


def tokenize(text):
    return re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))


def train_phrases():
    data = dict(NOTEBOOK_TRAIN)
    data.update(EXTRA_TRAIN)
    return data


def build_vocab(phrases):
    vocab = {UNK: 0}
    for text in phrases:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    words = tokenize(text)
    if not words:
        ids = [vocab[UNK]]
    else:
        ids = [vocab.get(word, vocab[UNK]) for word in words]
    return torch.LongTensor(ids).view(1, -1)


class MoodRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, token_ids):
        embedded = self.embed(token_ids)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])


def fit_mood(epochs=400, lr=1e-2, seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    phrases = train_phrases()
    vocab = build_vocab(phrases)
    model = MoodRNN(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    items = list(phrases.items())
    model.train()
    for _ in range(epochs):
        random.shuffle(items)
        for text, label in items:
            logits = model(encode(text, vocab))
            loss = criterion(logits, torch.tensor([int(label)]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model, vocab


def accuracy(model, vocab, phrases):
    correct = 0
    model.eval()
    with torch.no_grad():
        for text, label in phrases.items():
            logits = model(encode(text, vocab))
            pred = int(torch.argmax(logits, dim=-1))
            correct += int(pred == int(label))
    return correct / len(phrases)


def predict_mood(model, vocab, text):
    model.eval()
    with torch.no_grad():
        logits = model(encode(text, vocab))
        probs = torch.softmax(logits, dim=-1)[0]
    negative = float(probs[0])
    positive = float(probs[1])
    if max(positive, negative) < 0.6:
        label = "непонятно"
        score = max(positive, negative)
    elif positive >= negative:
        label = POSITIVE
        score = positive
    else:
        label = NEGATIVE
        score = negative
    return label, score


def mood_source(replied_text, inline_text):
    replied = (replied_text or "").strip()
    if replied:
        return replied
    return (inline_text or "").strip()


def save_mood(path, model, vocab, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.to("cpu")
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "embed_size": model.embed_size,
            "hidden_size": model.hidden_size,
            **meta,
        },
        path,
    )
    model.to(device)
    model.train(was_training)
    return path


def load_mood(path, device=None):
    if device is None:
        device = torch.device("cpu")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mood model file not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    vocab = payload["vocab"]
    model = MoodRNN(len(vocab), payload["embed_size"], payload["hidden_size"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    meta = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "vocab", "embed_size", "hidden_size"}
    }
    return model, vocab, meta
