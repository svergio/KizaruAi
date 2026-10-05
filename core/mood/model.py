import random
import re
from pathlib import Path

import torch
from torch import nn

from core.text.checkpoint import _torch_load_local

# notebooks/определенее настроение).ipynb — ось 0: 0 злой / 1 добрый; ось 1: 0 грустный / 1 веселый
VECTOR_TRAIN = {
    "добрый": (1, None),
    "злой": (0, None),
    "веселый": (None, 1),
    "грустный": (1, 0),
    "спокойный": (1, None),
    "раздраженный": (0, None),
    "радостный": (None, 1),
    "печальный": (1, 0),
    "круто": (1, 1),
    "радует": (1, 1),
    "ахуенно": (1, 1),
    "люблю": (1, 1),
    "кайф": (1, 1),
    "огонь": (1, 1),
    "класс": (1, 1),
    "мне хорошо": (1, 1),
    "счастье": (1, 1),
    "мне весело": (1, 1),
    "я смеюсь": (1, 1),
    "хорошее настроение": (1, 1),
    "ты лучший": (1, 1),
    "обожаю": (1, 1),
    "ура": (1, 1),
    "тоска": (1, 0),
    "печально": (1, 0),
    "мне грустно": (1, 0),
    "жалко": (1, 0),
    "слезы": (1, 0),
    "одиноко": (1, 0),
    "скучаю": (1, 0),
    "пусто на душе": (1, 0),
    "добрый но тоскливо": (1, 0),
    "хуево": (0, 0),
    "плохо": (0, 0),
    "ненавижу": (0, 0),
    "бесит": (0, 0),
    "мне плохо": (0, 0),
    "отстой": (0, 0),
    "ужас": (0, 0),
    "мерзко": (0, 0),
    "злюсь": (0, 0),
    "достало все": (0, 0),
    "плохой день": (0, 0),
    "ярость": (0, None),
    "злой и веселый": (0, 1),
    "бесит но смешно": (0, 1),
    "злорадно": (0, 1),
    "злюсь и ржу": (0, 1),
    "вредный но веселый": (0, 1),
    "добрый и веселый": (1, 1),
    "добрый и грустный": (1, 0),
    "злой и грустный": (0, 0),
    "добрый но грустный": (1, 0),
    "злой но веселый": (0, 1),
    "он добрый и смеется": (1, 1),
    "она добрая но плачет": (1, 0),
    "он злится и грустит": (0, 0),
    "он злой но шутит": (0, 1),
    "не злой": (1, None),
    "не добрый": (0, None),
    "не грустный": (None, 1),
    "не веселый": (None, 0),
    "я не злой": (1, None),
    "мне не грустно": (None, 1),
    "совсем не весело": (None, 0),
}

VECTOR_DEMO = (
    "добрый",
    "злой",
    "грустный",
    "веселый",
    "круто",
    "хуево",
    "злой и грустный",
    "добрый но грустный",
    "злой но веселый",
    "ты добрый и веселый",
    "очень злой и грустный",
    "да иди нахуй",
)

UNK = "<unk>"
MOOD_CHECKPOINT_KIND = "vector2"


def tokenize(text):
    return re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))


def train_phrases():
    return dict(VECTOR_TRAIN)


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
    def __init__(self, vocab_size, embed_size=32, hidden_size=32, n_layers=2):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, token_ids):
        embedded = self.embed(token_ids)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])


def _vector_loss(logits, target):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    for axis, bit in enumerate(target):
        if bit is None:
            continue
        t = torch.tensor([float(bit)], dtype=logits.dtype, device=logits.device)
        losses.append(criterion(logits[0, axis : axis + 1], t))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat(losses).mean()


def fit_mood(epochs=300, lr=3e-3, seed=2):
    random.seed(seed)
    torch.manual_seed(seed)
    phrases = train_phrases()
    vocab = build_vocab(phrases)
    model = MoodRNN(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    items = list(phrases.items())
    model.train()
    for _ in range(epochs):
        random.shuffle(items)
        for text, target in items:
            logits = model(encode(text, vocab))
            loss = _vector_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model, vocab


def _axis_match(prob, bit):
    if bit is None:
        return True
    return (prob >= 0.5) == (float(bit) >= 0.5)


def accuracy(model, vocab, phrases):
    if not phrases:
        return 1.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for text, target in phrases.items():
            _, kind, _, fun = predict_mood(model, vocab, text)
            ok = _axis_match(kind, target[0]) and _axis_match(fun, target[1])
            correct += int(ok)
    return correct / len(phrases)


def axis_accuracy(model, vocab, phrases):
    counted = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for text, target in phrases.items():
            _, kind, _, fun = predict_mood(model, vocab, text)
            for axis, prob, bit in ((0, kind, target[0]), (1, fun, target[1])):
                if bit is None:
                    continue
                counted += 1
                if _axis_match(prob, bit):
                    correct += 1
    return correct / max(counted, 1)


def predict_mood(model, vocab, text):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(encode(text, vocab).to(device))
        probs = torch.sigmoid(logits)[0]
    kind = float(probs[0])
    fun = float(probs[1])
    kind_name = "добрый" if kind >= 0.5 else "злой"
    fun_name = "веселый" if fun >= 0.5 else "грустный"
    return kind_name, kind, fun_name, fun


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
            "n_layers": model.n_layers,
            "kind": MOOD_CHECKPOINT_KIND,
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
    payload = _torch_load_local(path, device)
    if payload.get("kind") != MOOD_CHECKPOINT_KIND:
        raise ValueError(
            "Unsupported mood checkpoint (expected 2D vector model). "
            "Retrain with: python train_mood.py --out checkpoints/mood.pt"
        )
    vocab = payload["vocab"]
    model = MoodRNN(
        len(vocab),
        payload["embed_size"],
        payload["hidden_size"],
        n_layers=payload.get("n_layers", 2),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    meta = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "vocab", "embed_size", "hidden_size", "n_layers", "kind"}
    }
    return model, vocab, meta
