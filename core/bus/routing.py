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

# Four algorithms, four mode_keys — not a details flag on line/render.
PIC_MODE_KEYS = {
    "line": "1",
    "simple": "2",
    "render": "3",
    "details": "4",
}
PIC_KEY_TO_NAME = {key: name for name, key in PIC_MODE_KEYS.items()}


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
    data: bytes | None,
    filename: str,
    *,
    scale: int | None = None,
    cells: int | None = None,
    animated: bool | None = None,
    source_path: str | None = None,
    out_path: str | None = None,
    line: dict | None = None,
    as_file: bool | None = None,
    dark: str | None = None,
    light: str | None = None,
    line_thr: float | None = None,
    details_thr: float | None = None,
    smile_px: int | None = None,
    upscale: bool | None = None,
) -> dict:
    if mode_key not in PIC_KEY_TO_NAME:
        raise ValueError(f"unknown pic mode_key: {mode_key}")
    if not source_path and data is None:
        raise ValueError("pic payload needs data or source_path")
    mode_name = PIC_KEY_TO_NAME[mode_key]
    payload = {
        "mode_key": mode_key,
        "filename": filename or "input.bin",
    }
    if source_path:
        payload["source_path"] = str(source_path)
    else:
        payload["data_b64"] = base64.b64encode(data).decode("ascii")
    if out_path:
        payload["out_path"] = str(out_path)
    # render always files; Discord `cli` / CLI out_path → as_file.
    want_file = bool(out_path) if as_file is None else bool(as_file)
    if mode_name == "render":
        want_file = True if as_file is None else bool(as_file)
    if want_file:
        payload["as_file"] = True
    if mode_name == "simple":
        from core.image.ascii import resolve_simple_glyphs

        glyphs = resolve_simple_glyphs(dark, light)
        payload["dark"] = glyphs["dark"]
        payload["light"] = glyphs["light"]
    if want_file or mode_name == "render":
        if scale is not None:
            payload["scale"] = max(1, int(scale))
        if cells is not None:
            payload["cells"] = int(cells)
        if animated is not None:
            payload["animated"] = bool(animated)
        if smile_px is not None:
            payload["smile_px"] = max(1, int(smile_px))
        if upscale is not None:
            payload["upscale"] = bool(upscale)
    if mode_name in ("line", "render", "details"):
        from core.image.ascii import _mode_line_defaults

        line_payload = _mode_line_defaults(
            mode_name,
            line=line,
            line_thr=line_thr,
            details_thr=details_thr,
        )
        # Persist only the knobs that differ from pure defaults / identify the mode.
        slim = {}
        if mode_name == "render":
            slim["tone"] = line_payload["tone"]
            slim["structure"] = line_payload["structure"]
        if mode_name == "details":
            slim["details"] = True
        else:
            slim["details"] = False
        if line_thr is not None:
            slim["line_thr"] = line_payload["line_thr"]
        if details_thr is not None:
            slim["details_thr"] = line_payload["details_thr"]
        if line:
            for key in ("tone", "structure", "threshold", "min_size", "contrast", "step", "depth"):
                if key in line:
                    slim[key] = line_payload[key]
        payload["line"] = slim
    return payload


def pic_mode_name_to_key(mode: str, *, file_output: bool = False) -> str:
    from core.image.ascii import resolve_mode

    del file_output  # mode_key is the algorithm; as_file is a separate payload flag
    return PIC_MODE_KEYS[resolve_mode(mode)]


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
        if mode_key not in PIC_KEY_TO_NAME:
            raise ValueError(f"unknown pic mode: {mode_key}")
        mode_name = PIC_KEY_TO_NAME[mode_key]
        as_file = bool(payload.get("as_file") or payload.get("out_path") or mode_name == "render")
        if "animated" in payload:
            if payload["animated"] and as_file and mode_name != "simple":
                return "pic_video"
            return "pic_still"
        if as_file and mode_name != "simple":
            filename = payload.get("filename") or ""
            source_path = payload.get("source_path")
            if source_path:
                from pathlib import Path

                path = Path(source_path)
                data = path.read_bytes() if path.is_file() else b""
                filename = filename or path.name
            else:
                data = base64.b64decode(payload["data_b64"])
            return "pic_video" if is_animated_media(data, filename) else "pic_still"
        return "pic_still"
    raise ValueError(f"unknown bot kind: {bot_kind}")
