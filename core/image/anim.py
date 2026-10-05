import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.image.ascii import CLI_DEFAULT_CELLS, CLI_DEFAULT_SCALE, resolve_mode
from core.image.render import GPU_BATCH, pick_device, render_rgb_batch

ANIM_SUFFIXES = {".gif", ".mp4", ".webm", ".mkv", ".mov", ".avi"}
ANIM_DEFAULT_CELLS = CLI_DEFAULT_CELLS
ANIM_DEFAULT_FPS = None
ANIM_DEFAULT_SCALE = CLI_DEFAULT_SCALE


def is_anim_source(path):
    return Path(path).suffix.lower() in ANIM_SUFFIXES


def require_tool(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found on PATH")


def cells_for_source(width):
    # Число колонок — --cells (320 по умолчанию), не ширина источника / 4.
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


def probe_video(source):
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
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
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
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
    mode_name = resolve_mode(mode)
    device = pick_device()
    batch_size = GPU_BATCH if device.type == "cuda" else 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(
        f"anim device={device} batch={batch_size} cells={cells} scale={scale} "
        f"fps={fps:.3f} source={meta['width']}x{meta['height']}",
        flush=True,
    )
    frame_bytes = meta["width"] * meta["height"] * 3
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = None
    gif_frames = []
    done = 0
    last_report = 0
    total = meta["frames"]
    started = time.perf_counter()
    out_h = out_w = None
    try:
        while True:
            batch = []
            for _ in range(batch_size):
                raw = _read_exact(decoder.stdout, frame_bytes)
                if raw is None:
                    break
                batch.append(np.frombuffer(raw, dtype=np.uint8).reshape(meta["height"], meta["width"], 3))
            if not batch:
                break
            stacked = np.stack(batch, axis=0)
            rendered = render_rgb_batch(
                stacked,
                mode=mode_name,
                cells=cells,
                discord=discord,
                scale=scale,
                device=device,
                cap=False,
                upscale=upscale,
                line=line,
            )
            if encoder is None:
                out_h, out_w = int(rendered.shape[-2]), int(rendered.shape[-1])
                encoder = _open_encoder(out_path, out_w, out_h, fps, source if audio else None)
            for frame in rendered.cpu().numpy():
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                if gif_path is not None:
                    gif_frames.append(Image.fromarray(frame, mode="L"))
                done += 1
            if last_report == 0 or done - last_report >= 25 or len(batch) < batch_size:
                label = f"{done}/{total}" if total else str(done)
                print(f"anim {label}", flush=True)
                last_report = done
        if encoder is None:
            raise RuntimeError(f"no frames decoded from {source}")
        encoder.stdin.close()
        enc_err = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(enc_err.strip() or "ffmpeg failed to encode mp4")
        dec_err = decoder.stderr.read().decode("utf-8", errors="replace")
        if decoder.wait() != 0:
            raise RuntimeError(dec_err.strip() or "ffmpeg failed to decode")
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
        if decoder.poll() is None:
            decoder.kill()
    if gif_path is not None:
        _frames_to_gif(gif_frames, gif_path, fps)
    elapsed = max(time.perf_counter() - started, 1e-6)
    peak = ""
    if device.type == "cuda":
        peak = f" vram_peak_mb={torch.cuda.max_memory_allocated() / (1024 * 1024):.1f}"
    print(
        f"anim done frames={done} sec={elapsed:.1f} throughput_fps={done / elapsed:.1f} "
        f"out={out_w}x{out_h}{peak}",
        flush=True,
    )
    return Path(out_path)


def _open_encoder(out_path, width, height, fps, audio_source):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        stream_anim(
            source,
            Path(tmp) / "line.mp4",
            mode=mode,
            cells=cells,
            fps=fps,
            scale=scale,
            discord=discord,
            audio=False,
            gif_path=out_path,
            upscale=upscale,
            line=line,
        )
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
