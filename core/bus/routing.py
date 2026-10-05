import base64

from core.image.probe import is_animated_media

WORKER_KINDS = (
    "text_lstm",
    "text_lstm_word",
    "text_transformer",
    "text_gpt",
    "text_deepseek",
    "text_bert_gen",
    "mood",
    "pic_still",
    "pic_video",
)

BOT_KINDS = ("text", "mood", "pic")

TEXT_MODELS = ("lstm", "lstm-word", "transformer", "gpt", "deepseek", "bert-gen")

WORKER_TO_BOT_MODEL = {
    "text_lstm": "lstm",
    "text_lstm_word": "lstm-word",
    "text_transformer": "transformer",
    "text_gpt": "gpt",
    "text_deepseek": "deepseek",
    "text_bert_gen": "bert-gen",
}


def build_text_payload(
    model: str,
    start: str = "Я",
    *,
    temp: float | None = None,
    length: int | None = None,
    lines: int | None = None,
    iters: int | None = None,
) -> dict:
    if model not in TEXT_MODELS:
        raise ValueError(f"unknown text model: {model}")
    payload = {"model": model, "start": start}
    for key, value in (("temp", temp), ("length", length), ("lines", lines), ("iters", iters)):
        if value is not None:
            payload[key] = value
    return payload


def build_pic_payload(
    mode_key: str,
    data: bytes,
    filename: str,
    *,
    scale: int | None = None,
    cells: int | None = None,
    animated: bool | None = None,
) -> dict:
    if mode_key not in ("1", "2", "3"):
        raise ValueError(f"unknown pic mode_key: {mode_key}")
    payload = {
        "mode_key": mode_key,
        "filename": filename or "input.bin",
        "data_b64": base64.b64encode(data).decode("ascii"),
    }
    if mode_key == "3":
        if scale is not None:
            payload["scale"] = max(1, int(scale))
        if cells is not None:
            payload["cells"] = int(cells)
        if animated is not None:
            payload["animated"] = bool(animated)
    return payload


def pic_mode_name_to_key(mode: str) -> str:
    from core.image.ascii import resolve_mode

    return "1" if resolve_mode(mode) == "line" else "2"


def resolve_worker_kind(bot_kind: str, payload: dict) -> str:
    if bot_kind == "text":
        model = payload.get("model", "lstm")
        if model not in TEXT_MODELS:
            raise ValueError(f"unknown text model: {model}")
        if model == "lstm":
            return "text_lstm"
        if model == "lstm-word":
            return "text_lstm_word"
        if model == "bert-gen":
            return "text_bert_gen"
        return f"text_{model}"
    if bot_kind == "mood":
        return "mood"
    if bot_kind == "pic":
        mode_key = payload.get("mode_key")
        if mode_key in ("1", "2"):
            return "pic_still"
        if mode_key == "3":
            if "animated" in payload:
                return "pic_video" if payload["animated"] else "pic_still"
            data = base64.b64decode(payload["data_b64"])
            filename = payload.get("filename") or ""
            return "pic_video" if is_animated_media(data, filename) else "pic_still"
        raise ValueError(f"unknown pic mode: {mode_key}")
    raise ValueError(f"unknown bot kind: {bot_kind}")
