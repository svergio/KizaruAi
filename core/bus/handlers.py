import base64
import tempfile
from io import BytesIO
from pathlib import Path

from core.image.anim import render_anim_gif
from core.image.ascii import (
    CLI_DEFAULT_CELLS,
    CLI_DEFAULT_SCALE,
    CLI_MAX_COLS,
    DISCORD_SMALL_CELLS,
    clamp_cells,
    discord_chunks,
    render_pic,
    render_png,
    resolve_mode,
)
from core.text.bert_gen import generate_mlm
from core.text.checkpoint import cuda_for_job, release_cuda, resolve_infer_device
from core.text.params import (
    WORD_LENGTH_MAX,
    WORD_LENGTH_MIN,
    resolve_text_params,
)
from core.text.sample import evaluate, evaluate_lm, evaluate_lm_words, evaluate_words

DISCORD_LIMIT = 1800
DISCORD_FILE_LIMIT = 8 * 1024 * 1024


def job_text_lstm(payload: dict, runtime) -> dict:
    start = payload.get("start", "Я")
    params = resolve_text_params("lstm", payload)
    model = runtime["lstm_model"]
    device = resolve_infer_device(runtime)
    with cuda_for_job(model, device):
        text = evaluate(
            model,
            runtime["char_to_idx"],
            runtime["idx_to_char"],
            start_text=start,
            prediction_len=params["length"],
            temp=params["temp"],
            device=device,
            max_lines=params["lines"],
        )
    return {"type": "message", "content": text[:DISCORD_LIMIT]}


def job_text_lstm_word(payload: dict, runtime) -> dict:
    start = payload.get("start", "Я")
    params = resolve_text_params("lstm-word", payload)
    model = runtime["lstm_model"]
    device = resolve_infer_device(runtime)
    with cuda_for_job(model, device):
        text = evaluate_words(
            model,
            runtime["token_to_idx"],
            runtime["idx_to_token"],
            start_text=start,
            prediction_len=params["length"],
            temp=params["temp"],
            device=device,
            max_lines=params["lines"],
        )
    return {"type": "message", "content": text[:DISCORD_LIMIT]}


def _job_text_lm(payload: dict, runtime, model_key: str, runtime_key: str) -> dict:
    if runtime.get(runtime_key) is None:
        raise RuntimeError(f"{model_key} not loaded")
    start = payload.get("start", "Я")
    params = resolve_text_params(model_key, payload)
    model, chars, idx = runtime[runtime_key]
    device = resolve_infer_device(runtime)
    level = runtime.get("level") or "char"
    with cuda_for_job(model, device):
        if level == "word":
            if payload.get("length") is None:
                length = 72
            else:
                length = max(WORD_LENGTH_MIN, min(WORD_LENGTH_MAX, int(payload["length"])))
            text = evaluate_lm_words(
                model,
                chars,
                idx,
                start_text=start,
                prediction_len=length,
                temp=params["temp"],
                device=device,
                max_lines=params["lines"],
            )
        else:
            text = evaluate_lm(
                model,
                chars,
                idx,
                start_text=start,
                prediction_len=params["length"],
                temp=params["temp"],
                device=device,
                max_lines=params["lines"],
            )
    return {"type": "message", "content": text[:DISCORD_LIMIT]}


def job_text_transformer(payload: dict, runtime) -> dict:
    return _job_text_lm(payload, runtime, "transformer", "transformer")


def job_text_gpt(payload: dict, runtime) -> dict:
    return _job_text_lm(payload, runtime, "gpt", "gpt")


def job_text_deepseek(payload: dict, runtime) -> dict:
    return _job_text_lm(payload, runtime, "deepseek", "deepseek")


def job_text_bert_gen(payload: dict, runtime) -> dict:
    if runtime.get("bert_gen") is None:
        raise RuntimeError("bert-gen not loaded")
    start = payload.get("start", "Я")
    params = resolve_text_params("bert-gen", payload)
    model, chars, idx = runtime["bert_gen"]
    device = resolve_infer_device(runtime)
    level = runtime.get("level") or "char"
    if level == "word":
        if payload.get("length") is None:
            seq_len = min(64, model.block_size)
        else:
            seq_len = max(WORD_LENGTH_MIN, min(WORD_LENGTH_MAX, int(payload["length"])))
    else:
        seq_len = params["length"]
    with cuda_for_job(model, device):
        text = generate_mlm(
            model,
            chars,
            idx,
            start_text=start,
            seq_len=min(seq_len, model.block_size),
            n_iters=params["iters"],
            temp=params["temp"],
            device=device,
            max_lines=params["lines"],
            level=level,
        )
    return {"type": "message", "content": text[:DISCORD_LIMIT]}


def job_mood(payload: dict, runtime) -> dict:
    if runtime["mood"] is None:
        raise RuntimeError("mood not loaded")
    from core.mood.model import predict_mood

    mood_model, vocab = runtime["mood"]
    source = payload["source"]
    device = resolve_infer_device(runtime)
    with cuda_for_job(mood_model, device):
        kind_name, kind, fun_name, fun = predict_mood(mood_model, vocab, source)
    snippet = " ".join(source.split())
    if len(snippet) > 140:
        snippet = snippet[:140] + "..."
    header = (
        f"Настроение: {kind_name} {kind:.2f} | {fun_name} {fun:.2f}\n"
        f"vector=({kind:.2f}, {fun:.2f})"
    )
    content = f"{header}\n{snippet}"[:DISCORD_LIMIT]
    return {"type": "message", "content": content}


def job_pic_still(payload: dict, _runtime) -> dict:
    data = base64.b64decode(payload["data_b64"])
    filename = payload.get("filename") or "input.bin"
    mode_key = payload["mode_key"]
    try:
        if mode_key == "3":
            scale_val = max(1, int(payload.get("scale", CLI_DEFAULT_SCALE)))
            cells_val = clamp_cells(int(payload.get("cells", CLI_DEFAULT_CELLS)), max_cells=CLI_MAX_COLS)
            with tempfile.TemporaryDirectory(prefix="kizaru_ds_") as tmp:
                root = Path(tmp)
                suffix = Path(filename).suffix.lower() or ".png"
                source = root / f"source{suffix}"
                source.write_bytes(data)
                png = render_png(
                    source,
                    mode="line",
                    cells=cells_val,
                    discord=False,
                    scale=scale_val,
                    upscale=True,
                )
                buf = BytesIO()
                png.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
                name = f"line_s{scale_val}.png"
                kind = f"png {png.size[0]}x{png.size[1]}"
            if len(raw) > DISCORD_FILE_LIMIT:
                return {
                    "type": "message",
                    "content": (
                        f"Файл слишком большой ({len(raw) // 1024} KB). "
                        "Уменьши cells, например: /ai pic mode:3 scale:2 cells:160"
                    ),
                }
            return {
                "type": "file",
                "content": f"cli line cells={cells_val} scale={scale_val} {kind}",
                "filename": name,
                "data_b64": base64.b64encode(raw).decode("ascii"),
            }

        mode_name = resolve_mode(mode_key)
        text = render_pic(data, mode=mode_name, cells=DISCORD_SMALL_CELLS, discord=True)
        chunks = discord_chunks(text)
        return {"type": "chunks", "chunks": chunks}
    finally:
        release_cuda()


def job_pic_video(payload: dict, _runtime) -> dict:
    data = base64.b64decode(payload["data_b64"])
    filename = payload.get("filename") or "input.bin"
    mode_key = payload.get("mode_key")
    if mode_key != "3":
        raise RuntimeError("pic_video only handles mode 3 animated")
    scale_val = max(1, int(payload.get("scale", CLI_DEFAULT_SCALE)))
    cells_val = clamp_cells(int(payload.get("cells", CLI_DEFAULT_CELLS)), max_cells=CLI_MAX_COLS)
    try:
        with tempfile.TemporaryDirectory(prefix="kizaru_ds_") as tmp:
            root = Path(tmp)
            suffix = Path(filename).suffix.lower() or ".gif"
            source = root / f"source{suffix}"
            source.write_bytes(data)
            out = root / f"line_s{scale_val}.gif"
            render_anim_gif(
                source,
                out,
                mode="line",
                cells=cells_val,
                scale=scale_val,
                discord=False,
                upscale=True,
            )
            raw = out.read_bytes()
            name = out.name
        if len(raw) > DISCORD_FILE_LIMIT:
            return {
                "type": "message",
                "content": (
                    f"Файл слишком большой ({len(raw) // 1024} KB). "
                    "Уменьши cells, например: /ai pic mode:3 scale:2 cells:160"
                ),
            }
        return {
            "type": "file",
            "content": f"cli line cells={cells_val} scale={scale_val} gif",
            "filename": name,
            "data_b64": base64.b64encode(raw).decode("ascii"),
        }
    finally:
        release_cuda()
