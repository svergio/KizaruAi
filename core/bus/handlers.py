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
from core.text.sample import (
    evaluate,
    evaluate_lm,
    evaluate_lm_bpe,
    evaluate_lm_words,
    evaluate_words,
)

DISCORD_LIMIT = 1800
DISCORD_FILE_LIMIT = 8 * 1024 * 1024


def _pic_input(payload: dict) -> tuple[bytes, str]:
    source_path = payload.get("source_path")
    if source_path:
        path = Path(source_path)
        if not path.is_file():
            raise RuntimeError(f"source not found: {path}")
        return path.read_bytes(), payload.get("filename") or path.name
    if "data_b64" not in payload:
        raise RuntimeError("pic payload missing data_b64/source_path")
    return base64.b64decode(payload["data_b64"]), payload.get("filename") or "input.bin"


def _pic_file_result(
    *,
    raw: bytes,
    name: str,
    content: str,
    out_path: str | None,
) -> dict:
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {
            "type": "file",
            "content": content,
            "filename": name,
            "path": str(path.resolve()),
        }
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
        "content": content,
        "filename": name,
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


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
        if level == "bpe":
            bpe_tok = runtime.get("bpe_tokenizer")
            if bpe_tok is None:
                raise RuntimeError("bpe tokenizer not loaded")
            if payload.get("length") is None:
                length = 72
            else:
                length = max(WORD_LENGTH_MIN, min(WORD_LENGTH_MAX, int(payload["length"])))
            text = evaluate_lm_bpe(
                model,
                bpe_tok,
                start_text=start,
                prediction_len=length,
                temp=params["temp"],
                device=device,
                max_lines=params["lines"],
            )
        elif level == "word":
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
    if runtime.get("deepseek") is None:
        raise RuntimeError("deepseek not loaded")
    from core.text.deepseek import generate as deepseek_generate

    start = payload.get("start", "Я")
    params = resolve_text_params("deepseek", payload)
    model, chars, idx = runtime["deepseek"]
    device = resolve_infer_device(runtime)
    level = runtime.get("level") or "char"
    if level == "word":
        if payload.get("length") is None:
            length = 72
        else:
            length = max(WORD_LENGTH_MIN, min(WORD_LENGTH_MAX, int(payload["length"])))
    else:
        length = params["length"]
    with cuda_for_job(model, device):
        text = deepseek_generate(
            model,
            chars,
            idx,
            start_text=start,
            prediction_len=length,
            temp=params["temp"],
            device=device,
            max_lines=params["lines"],
            level=level,
        )
    return {"type": "message", "content": text[:DISCORD_LIMIT]}


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
    from core.bus.cancel_flag import check_cancelled

    mode_key = payload["mode_key"]
    mode_name = resolve_mode(mode_key)
    out_path = payload.get("out_path")
    source_path = payload.get("source_path")
    as_file = bool(payload.get("as_file") or out_path or mode_name == "render")
    simple = None
    if mode_name == "simple":
        simple = {"dark": payload.get("dark"), "light": payload.get("light")}
    try:
        check_cancelled()
        if as_file:
            scale_val = max(1, int(payload.get("scale", CLI_DEFAULT_SCALE)))
            cells_val = clamp_cells(
                int(payload.get("cells", CLI_DEFAULT_CELLS)), max_cells=CLI_MAX_COLS
            )
            with tempfile.TemporaryDirectory(prefix="kizaru_ds_") as tmp:
                if source_path:
                    source = Path(source_path)
                    if not source.is_file():
                        raise RuntimeError(f"source not found: {source}")
                else:
                    data, filename = _pic_input(payload)
                    suffix = Path(filename).suffix.lower() or ".png"
                    source = Path(tmp) / f"source{suffix}"
                    source.write_bytes(data)
                smile_px = payload.get("smile_px")
                smile_px = None if smile_px is None else max(1, int(smile_px))
                upscale = bool(payload.get("upscale", True))
                png = render_png(
                    source,
                    mode=mode_name,
                    cells=cells_val,
                    discord=False,
                    scale=scale_val,
                    upscale=upscale,
                    line=payload.get("line"),
                    simple=simple,
                    smile_px=smile_px,
                )
                check_cancelled()
                font_tag = f" font={smile_px}" if smile_px is not None else ""
                if out_path:
                    path = Path(out_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    png.save(path, format="PNG", optimize=True)
                    name = path.name
                    return {
                        "type": "file",
                        "content": (
                            f"{mode_name} cells={cells_val} scale={scale_val}"
                            f"{font_tag} png {png.size[0]}x{png.size[1]}"
                        ),
                        "filename": name,
                        "path": str(path.resolve()),
                    }
                buf = BytesIO()
                png.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
                name = f"{mode_name}_s{scale_val}.png"
                kind = f"png {png.size[0]}x{png.size[1]}"
            return _pic_file_result(
                raw=raw,
                name=name,
                content=f"{mode_name} cells={cells_val} scale={scale_val}{font_tag} {kind}",
                out_path=None,
            )

        data, _filename = _pic_input(payload)
        text = render_pic(
            data,
            mode=mode_name,
            cells=DISCORD_SMALL_CELLS,
            discord=True,
            line=payload.get("line"),
            simple=simple,
        )
        check_cancelled()
        chunks = discord_chunks(text)
        return {"type": "chunks", "chunks": chunks}
    finally:
        release_cuda()


def job_pic_video(payload: dict, _runtime) -> dict:
    from core.bus.cancel_flag import check_cancelled

    mode_key = payload.get("mode_key")
    mode_name = resolve_mode(mode_key)
    if mode_name == "simple":
        raise RuntimeError("pic_video does not support simple mode")
    if mode_name not in ("line", "render", "details"):
        raise RuntimeError(f"pic_video unsupported mode: {mode_name}")
    scale_val = max(1, int(payload.get("scale", CLI_DEFAULT_SCALE)))
    cells_val = clamp_cells(int(payload.get("cells", CLI_DEFAULT_CELLS)), max_cells=CLI_MAX_COLS)
    out_path = payload.get("out_path")
    source_path = payload.get("source_path")
    try:
        check_cancelled()
        with tempfile.TemporaryDirectory(prefix="kizaru_ds_") as tmp:
            root = Path(tmp)
            if source_path:
                source = Path(source_path)
                if not source.is_file():
                    raise RuntimeError(f"source not found: {source}")
            else:
                data, filename = _pic_input(payload)
                suffix = Path(filename).suffix.lower() or ".gif"
                source = root / f"source{suffix}"
                source.write_bytes(data)
            out = Path(out_path) if out_path else root / f"{mode_name}_s{scale_val}.gif"
            if out_path:
                out.parent.mkdir(parents=True, exist_ok=True)
            render_anim_gif(
                source,
                out,
                mode=mode_name,
                cells=cells_val,
                scale=scale_val,
                discord=False,
                upscale=True,
                line=payload.get("line"),
            )
            check_cancelled()
            name = out.name
            if out_path:
                return {
                    "type": "file",
                    "content": f"{mode_name} cells={cells_val} scale={scale_val} gif",
                    "filename": name,
                    "path": str(out.resolve()),
                }
            raw = out.read_bytes()
        return _pic_file_result(
            raw=raw,
            name=name,
            content=f"{mode_name} cells={cells_val} scale={scale_val} gif",
            out_path=None,
        )
    finally:
        release_cuda()
