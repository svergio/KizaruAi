import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from core.image.ascii import CLI_DEFAULT_CELLS, CLI_DEFAULT_SCALE, resolve_mode
from core.image.mask_pool import (
    close_mask_pool,
    collect_line_masks,
    ensure_mask_pool,
    submit_line_masks,
)
from core.image.render import (
    pick_device,
    pick_gpu_batch,
    prepare_mask_source,
    render_rgb_batch,
    _patterns_tensor,
)

ANIM_SUFFIXES = {".gif", ".mp4", ".webm", ".mkv", ".mov", ".avi"}
ANIM_DEFAULT_CELLS = CLI_DEFAULT_CELLS
ANIM_DEFAULT_FPS = None
ANIM_DEFAULT_SCALE = CLI_DEFAULT_SCALE
_VAR_DIR = Path(__file__).resolve().parents[2] / "var"


def is_anim_source(path):
    return Path(path).suffix.lower() in ANIM_SUFFIXES


def require_tool(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found on PATH")


def cells_for_source(width):
    # Число колонок задаёт --cells (по умолчанию 320), не ширина источника / 4.
    del width
    return CLI_DEFAULT_CELLS


def _parse_rate(text):
    if not text or text in {"0/0", "N/A"}:
        return 30.0
    if "/" in str(text):
        num, den = str(text).split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 30.0
    return float(text)


def _stream_rotation(stream):
    tags = stream.get("tags") or {}
    rotate = tags.get("rotate")
    if rotate is not None and str(rotate).strip() != "":
        try:
            return float(rotate)
        except (TypeError, ValueError):
            pass
    for side in stream.get("side_data_list") or []:
        if "rotation" not in side:
            continue
        try:
            return float(side["rotation"])
        except (TypeError, ValueError):
            continue
    return 0.0


def _dims_after_rotation(width, height, rotation):
    angle = abs(float(rotation)) % 360.0
    if angle in (90.0, 270.0):
        return int(height), int(width)
    return int(width), int(height)


def _even_dims(width, height):
    width = int(width)
    height = int(height)
    return width + (width & 1), height + (height & 1)


def probe_video(source):
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(source),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    stream = json.loads(proc.stdout)["streams"][0]
    frames = stream.get("nb_frames")
    try:
        frame_count = int(frames) if frames not in (None, "N/A") else None
    except ValueError:
        frame_count = None
    width, height = _dims_after_rotation(
        int(stream["width"]),
        int(stream["height"]),
        _stream_rotation(stream),
    )
    return {
        "width": width,
        "height": height,
        "fps": _parse_rate(stream.get("avg_frame_rate")),
        "frames": frame_count,
    }


def _read_exact(stream, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _ffmpeg_rate(fps):
    text = f"{float(fps):.6f}".rstrip("0").rstrip(".")
    return text or "30"


def _new_render_id():
    return uuid.uuid4().hex


def _stderr_log_path(label, render_id):
    return _VAR_DIR / f"anim_ffmpeg_{label}_{os.getpid()}_{render_id}_stderr.log"


def _start_stderr_drain(proc, label, render_id):
    chunks = []
    log_path = _stderr_log_path(label, render_id)

    def _read():
        try:
            while True:
                block = proc.stderr.read(65536)
                if not block:
                    break
                chunks.append(block)
        except Exception:
            pass
        try:
            _VAR_DIR.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(b"".join(chunks))
        except OSError:
            pass

    thread = threading.Thread(target=_read, name=f"ffmpeg-stderr-{label}", daemon=True)
    thread.start()
    return thread, chunks, log_path


def _stderr_text(chunks):
    return b"".join(chunks).decode("utf-8", errors="replace")


def _cleanup_stderr_logs(paths):
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _ffmpeg_fail(chunks, log_path, fallback):
    detail = _stderr_text(chunks).strip() or fallback
    if log_path is not None:
        print(f"anim ffmpeg stderr kept: {log_path}", flush=True)
        detail = f"{detail}\nstderr log: {log_path}"
    raise RuntimeError(detail)


def encode_mp4_from_pngs(png_paths, out_path, fps=ANIM_DEFAULT_FPS, audio_source=None):
    require_tool("ffmpeg")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not png_paths:
        raise RuntimeError("no png frames to encode")
    parent = Path(png_paths[0]).parent
    pattern = parent / "frame_%06d.png"
    fps = 30 if fps is None else max(1, int(round(float(fps))))
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
    ]
    if audio_source is not None:
        cmd.extend(["-i", str(audio_source)])
    cmd.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if audio_source is not None:
        cmd.extend(["-map", "1:a:0?", "-shortest"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed to encode mp4")
    return out_path


def _needs_file_line_mask(mode_name, discord, _line):
    return (not discord) and mode_name == "line"


def _process_rendered_batch(
    rendered,
    state,
    *,
    out_path,
    fps,
    audio_source,
    batch_len,
    batch_size,
    total,
    render_id,
):
    if state["encoder"] is None:
        src_h, src_w = int(rendered.shape[-2]), int(rendered.shape[-1])
        out_w, out_h = _even_dims(src_w, src_h)
        state["out_w"] = out_w
        state["out_h"] = out_h
        state["src_h"] = src_h
        state["src_w"] = src_w
        state["encoder"] = _open_encoder(out_path, out_w, out_h, fps, audio_source)
        (
            state["enc_stderr_thread"],
            state["enc_stderr_chunks"],
            state["enc_stderr_log"],
        ) = _start_stderr_drain(state["encoder"], "encoder", render_id)
        state["frame_buf"] = np.zeros((out_h, out_w), dtype=np.uint8)
    frame_buf = state["frame_buf"]
    src_h = state["src_h"]
    src_w = state["src_w"]
    for index in range(int(rendered.shape[0])):
        frame = rendered[index].detach().to("cpu", non_blocking=False).numpy()
        if frame.shape[0] != src_h or frame.shape[1] != src_w:
            src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
            state["src_h"], state["src_w"] = src_h, src_w
        if frame_buf.shape[0] == src_h and frame_buf.shape[1] == src_w:
            np.copyto(frame_buf, frame)
        else:
            frame_buf.fill(0)
            frame_buf[:src_h, :src_w] = frame
        state["encoder"].stdin.write(frame_buf.tobytes())
        state["done"] += 1
    done = state["done"]
    last_report = state["last_report"]
    if last_report == 0 or done - last_report >= 25 or batch_len < batch_size:
        label = f"{done}/{total}" if total else str(done)
        print(f"anim {label}", flush=True)
        state["last_report"] = done


def stream_anim(
    source,
    out_path,
    mode="line",
    cells=None,
    fps=None,
    scale=ANIM_DEFAULT_SCALE,
    discord=False,
    audio=False,
    gif_path=None,
    upscale=True,
    line=None,
):
    require_tool("ffmpeg")
    source = Path(source)
    meta = probe_video(source)
    if cells is None:
        cells = CLI_DEFAULT_CELLS
    if fps is None:
        fps = meta["fps"]
    from core.image.ascii import _mode_line_defaults, _paint_mode

    mode_name = resolve_mode(mode)
    if mode_name == "simple":
        raise ValueError("anim supports mode line|render|details only")
    line = _mode_line_defaults(mode_name, line=line)
    paint_mode = _paint_mode(mode_name)
    device = pick_device()
    batch_size = pick_gpu_batch(meta["height"], meta["width"], device)
    use_mask_pool = _needs_file_line_mask(paint_mode, discord, line) and device.type != "cuda"
    mask_workers = 0
    if use_mask_pool:
        _pool, mask_workers = ensure_mask_pool()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    patterns = _patterns_tensor(device) if device.type == "cuda" else None
    enc_tag = "nvenc" if _has_nvenc() else "x264"
    print(
        f"anim device={device} batch={batch_size} mask_workers={mask_workers} "
        f"encode={enc_tag} cells={cells} scale={scale} fps={fps:.3f} "
        f"source={meta['width']}x{meta['height']}",
        flush=True,
    )
    frame_bytes = meta["width"] * meta["height"] * 3
    fps_filter = _ffmpeg_rate(fps)
    render_id = _new_render_id()
    # Software decode: same full-res pixels; CUDA hwaccel on AV1/odd codecs can judder.
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={fps_filter}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dec_stderr_thread, dec_stderr_chunks, dec_stderr_log = _start_stderr_drain(
        decoder, "decoder", render_id
    )
    state = {
        "encoder": None,
        "enc_stderr_thread": None,
        "enc_stderr_chunks": [],
        "enc_stderr_log": None,
        "out_h": None,
        "out_w": None,
        "src_h": None,
        "src_w": None,
        "frame_buf": None,
        "done": 0,
        "last_report": 0,
    }
    pending = None
    total = meta["frames"]
    started = time.perf_counter()
    audio_source = source if audio else None
    success = False
    out_path = Path(out_path)

    def _flush_pending(item):
        from core.bus.cancel_flag import check_cancelled

        check_cancelled()
        rgb_tensor, mask_futures, batch_len = item
        masks_cpu = collect_line_masks(mask_futures) if mask_futures is not None else None
        rendered = render_rgb_batch(
            None,
            mode=mode_name,
            cells=cells,
            discord=discord,
            scale=scale,
            device=device,
            patterns=patterns,
            cap=False,
            upscale=upscale,
            line=line,
            masks_cpu=masks_cpu,
            rgb_tensor=rgb_tensor,
        )
        _process_rendered_batch(
            rendered,
            state,
            out_path=out_path,
            fps=fps,
            audio_source=audio_source,
            batch_len=batch_len,
            batch_size=batch_size,
            total=total,
            render_id=render_id,
        )

    def _kill_ffmpeg():
        encoder = state["encoder"]
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
        if decoder.poll() is None:
            decoder.kill()

    def _drop_incomplete_out():
        try:
            if out_path.is_file():
                out_path.unlink()
        except OSError:
            pass
        if gif_path is not None:
            try:
                gp = Path(gif_path)
                if gp.is_file():
                    gp.unlink()
            except OSError:
                pass

    try:
        from core.bus.cancel_flag import GenerationCancelled, check_cancelled

        while True:
            check_cancelled()
            batch = []
            for _ in range(batch_size):
                raw = _read_exact(decoder.stdout, frame_bytes)
                if raw is None:
                    break
                batch.append(
                    np.frombuffer(raw, dtype=np.uint8).reshape(
                        meta["height"], meta["width"], 3
                    )
                )
            if not batch:
                break
            stacked = np.stack(batch, axis=0)
            if use_mask_pool:
                prep = prepare_mask_source(
                    stacked,
                    mode=mode_name,
                    line=line,
                    device=device,
                    non_blocking=device.type == "cuda",
                )
                futures, _n = submit_line_masks(prep["mask_cpu"], prep["opts"])
                if pending is not None:
                    _flush_pending(pending)
                pending = (prep["rgb"], futures, len(batch))
            else:
                if pending is not None:
                    _flush_pending(pending)
                    pending = None
                rendered = render_rgb_batch(
                    stacked,
                    mode=mode_name,
                    cells=cells,
                    discord=discord,
                    scale=scale,
                    device=device,
                    patterns=patterns,
                    cap=False,
                    upscale=upscale,
                    line=line,
                )
                _process_rendered_batch(
                    rendered,
                    state,
                    out_path=out_path,
                    fps=fps,
                    audio_source=audio_source,
                    batch_len=len(batch),
                    batch_size=batch_size,
                    total=total,
                    render_id=render_id,
                )
        if pending is not None:
            _flush_pending(pending)
        encoder = state["encoder"]
        if encoder is None:
            raise RuntimeError(f"no frames decoded from {source}")
        encoder.stdin.close()
        enc_rc = encoder.wait()
        if state["enc_stderr_thread"] is not None:
            state["enc_stderr_thread"].join(timeout=120)
        if enc_rc != 0:
            _ffmpeg_fail(
                state["enc_stderr_chunks"],
                state["enc_stderr_log"],
                "ffmpeg failed to encode mp4",
            )
        dec_rc = decoder.wait()
        dec_stderr_thread.join(timeout=120)
        if dec_rc != 0:
            _ffmpeg_fail(dec_stderr_chunks, dec_stderr_log, "ffmpeg failed to decode")
        success = True
    except GenerationCancelled:
        _kill_ffmpeg()
        _drop_incomplete_out()
        raise
    finally:
        _kill_ffmpeg()
        if use_mask_pool:
            close_mask_pool()
        if success:
            _cleanup_stderr_logs([dec_stderr_log, state.get("enc_stderr_log")])
        else:
            _drop_incomplete_out()
    if gif_path is not None:
        _mp4_to_gif(out_path, gif_path)
    elapsed = max(time.perf_counter() - started, 1e-6)
    peak = ""
    if device.type == "cuda":
        peak = f" vram_peak_mb={torch.cuda.max_memory_allocated() / (1024 * 1024):.1f}"
    out_w, out_h = state["out_w"], state["out_h"]
    done = state["done"]
    print(
        f"anim done frames={done} sec={elapsed:.1f} throughput_fps={done / elapsed:.1f} "
        f"out={out_w}x{out_h} batch={batch_size} mask_workers={mask_workers}{peak}",
        flush=True,
    )
    return Path(out_path)


_NVENC = None


def _has_nvenc():
    global _NVENC
    if _NVENC is not None:
        return _NVENC
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        _NVENC = "h264_nvenc" in (proc.stdout or "")
    except OSError:
        _NVENC = False
    return _NVENC


def _open_encoder(out_path, width, height, fps, audio_source):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _even_dims(width, height)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        _ffmpeg_rate(fps),
        "-i",
        "pipe:0",
    ]
    if audio_source is not None:
        cmd.extend(["-i", str(audio_source)])
    cmd.extend(["-map", "0:v:0"])
    if audio_source is not None:
        cmd.extend(["-map", "1:a:0?", "-shortest"])
    # CFR out; avoid nvenc -tune ull (no B-frames → choppy playback).
    cmd.extend(["-r", _ffmpeg_rate(fps), "-fps_mode", "cfr"])
    if _has_nvenc():
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
                "-bf",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _frames_to_gif(frames, out_path, fps):
    from PIL import Image

    out_path = Path(out_path)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-3))))
    paletted = [frame.convert("P", palette=Image.ADAPTIVE, colors=64) for frame in frames]
    first, rest = paletted[0], paletted[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out_path


def _pngs_to_gif(pngs, out_path, fps=30):
    from PIL import Image

    out_path = Path(out_path)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(fps))
    duration_ms = max(1, int(round(1000.0 / fps)))
    mosaics = [Image.open(path).convert("P", palette=Image.ADAPTIVE, colors=64) for path in pngs]
    first, rest = mosaics[0], mosaics[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out_path


def render_anim_mp4(
    source,
    out_path,
    mode="line",
    cells=ANIM_DEFAULT_CELLS,
    fps=ANIM_DEFAULT_FPS,
    scale=ANIM_DEFAULT_SCALE,
    discord=False,
    audio=False,
    gif_path=None,
    upscale=True,
    line=None,
):
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    return stream_anim(
        source,
        out_path,
        mode=mode,
        cells=cells,
        fps=fps,
        scale=scale,
        discord=discord,
        audio=audio,
        gif_path=gif_path,
        upscale=upscale,
        line=line,
    )


def _mp4_to_gif(mp4_path, gif_path):
    require_tool("ffmpeg")
    mp4_path = Path(mp4_path)
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    palette = gif_path.with_suffix(".palette.png")
    try:
        gen = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(mp4_path),
                "-vf",
                "palettegen=stats_mode=diff",
                str(palette),
            ],
            capture_output=True,
            text=True,
        )
        if gen.returncode != 0:
            raise RuntimeError(gen.stderr.strip() or "ffmpeg palettegen failed")
        use = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(mp4_path),
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=5",
                str(gif_path),
            ],
            capture_output=True,
            text=True,
        )
        if use.returncode != 0:
            raise RuntimeError(use.stderr.strip() or "ffmpeg paletteuse failed")
    finally:
        if palette.exists():
            palette.unlink()
    return gif_path


def render_anim_gif(
    source,
    out_path,
    mode="line",
    cells=ANIM_DEFAULT_CELLS,
    fps=ANIM_DEFAULT_FPS,
    scale=ANIM_DEFAULT_SCALE,
    discord=False,
    upscale=True,
    line=None,
):
    import tempfile

    out_path = Path(out_path)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    with tempfile.TemporaryDirectory(prefix="kizaru_anim_") as tmp:
        mp4 = Path(tmp) / "line.mp4"
        stream_anim(
            source,
            mp4,
            mode=mode,
            cells=cells,
            fps=fps,
            scale=scale,
            discord=discord,
            audio=False,
            gif_path=None,
            upscale=upscale,
            line=line,
        )
        _mp4_to_gif(mp4, out_path)
    return out_path


def mux_anim_mp4(video_path, source_path, mp4_path):
    require_tool("ffmpeg")
    video_path = Path(video_path)
    source_path = Path(source_path)
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.suffix.lower() == ".mp4":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(mp4_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(mp4_path),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed to mux mp4")
    return mp4_path
