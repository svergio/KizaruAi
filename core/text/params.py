from core.bus.routing import TEXT_MODELS

TEMP_MIN = 0.1
TEMP_MAX = 2.0
LINES_MIN = 1
LINES_MAX = 12
ITERS_MIN = 1
ITERS_MAX = 200

CHAR_LENGTH_MIN = 40
CHAR_LENGTH_MAX = 1200
WORD_LENGTH_MIN = 8
WORD_LENGTH_MAX = 200

# Температуры подобраны по кривой "луп против мусора": docs/research/generation-notes.md
DEFAULTS = {
    "lstm": {"temp": 0.4, "length": 360, "lines": 6, "iters": 32},
    "lstm-word": {"temp": 0.8, "length": 72, "lines": 6, "iters": 32},
    "transformer": {"temp": 0.9, "length": 360, "lines": 6, "iters": 32},
    "gpt": {"temp": 1.0, "length": 360, "lines": 6, "iters": 32},
    "deepseek": {"temp": 0.9, "length": 360, "lines": 6, "iters": 32},
    "bert-gen": {"temp": 1.0, "length": 128, "lines": 6, "iters": 32},
}

WORD_LEVEL_MODELS = ("lstm-word",)

PARAM_KEYS = ("temp", "length", "lines", "iters")


def length_bounds(model: str) -> tuple[int, int]:
    if model in WORD_LEVEL_MODELS:
        return WORD_LENGTH_MIN, WORD_LENGTH_MAX
    return CHAR_LENGTH_MIN, CHAR_LENGTH_MAX


def _clamp(value, low, high):
    return max(low, min(high, value))


def resolve_text_params(model: str, payload: dict | None = None) -> dict:
    if model not in TEXT_MODELS:
        raise ValueError(f"unknown text model: {model}")
    payload = payload or {}
    defaults = DEFAULTS[model]
    out = dict(defaults)

    if payload.get("temp") is not None:
        out["temp"] = _clamp(float(payload["temp"]), TEMP_MIN, TEMP_MAX)
    if payload.get("lines") is not None:
        out["lines"] = _clamp(int(payload["lines"]), LINES_MIN, LINES_MAX)
    if payload.get("iters") is not None:
        out["iters"] = _clamp(int(payload["iters"]), ITERS_MIN, ITERS_MAX)
    if payload.get("length") is not None:
        low, high = length_bounds(model)
        out["length"] = _clamp(int(payload["length"]), low, high)

    return out


def collect_overrides(temp=None, length=None, lines=None, iters=None) -> dict:
    given = {"temp": temp, "length": length, "lines": lines, "iters": iters}
    return {key: value for key, value in given.items() if value is not None}


def describe(model: str, params: dict) -> str:
    parts = [f"model={model}", f"temp={params['temp']:g}", f"lines={params['lines']}"]
    unit = "tokens" if model in WORD_LEVEL_MODELS else "chars"
    if model == "bert-gen":
        parts.append(f"seq_len={params['length']}")
        parts.append(f"iters={params['iters']}")
    else:
        parts.append(f"length={params['length']} {unit}")
    return " ".join(parts)
