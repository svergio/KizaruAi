"""core/image/anim.py: видеорендер без реального ffmpeg/GPU/видео."""
import io
import json
import os
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.image import anim


@pytest.fixture(autouse=True)
def _reset_nvenc_cache():
    anim._NVENC = None
    yield
    anim._NVENC = None


@pytest.fixture(autouse=True)
def _anim_var_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(anim, "_VAR_DIR", tmp_path)
    return tmp_path


def _run_stream_anim(tmp_path, *, returncode=0, stderr_bytes=b"", name="out"):
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=stderr_bytes)
    encoder = _FakeProc(stderr_bytes=stderr_bytes, returncode=returncode)
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)

    def fake_popen(cmd, **kwargs):
        return decoder if "pipe:1" in cmd else encoder

    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=gray
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ):
        return anim.stream_anim(
            "src.mp4",
            tmp_path / f"{name}.mp4",
            mode="line",
            cells=1,
            discord=True,
        )


def _ffprobe_payload(width, height, fps="30/1", frames=2, rotation=None, rotate_tag=None):
    stream = {
        "width": width,
        "height": height,
        "avg_frame_rate": fps,
        "nb_frames": str(frames),
    }
    if rotation is not None:
        stream["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rotation}]
    if rotate_tag is not None:
        stream["tags"] = {"rotate": str(rotate_tag)}
    return json.dumps({"streams": [stream]})


class _FakePipe:
    def __init__(self, chunks=None, stderr_bytes=b""):
        self._buf = io.BytesIO(b"".join(chunks or []))
        self._stderr = io.BytesIO(stderr_bytes)
        self._writes = []

    def read(self, n=-1):
        return self._buf.read(n)

    def write(self, data):
        self._writes.append(bytes(data))
        return len(data)

    def close(self):
        return None


class _FakeProc:
    def __init__(self, stdout=None, stderr_bytes=b"", returncode=0):
        self.stdout = stdout
        self.stderr = io.BytesIO(stderr_bytes)
        self.stdin = _FakePipe()
        self.returncode = returncode
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self):
        self._alive = False
        return self.returncode

    def kill(self):
        self._alive = False


def test_probe_video_swaps_dims_on_rotation_90():
    payload = _ffprobe_payload(1920, 1080, rotation=90)
    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.subprocess.run"
    ) as run:
        run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
        meta = anim.probe_video("clip.mp4")
    assert meta["width"] == 1080
    assert meta["height"] == 1920
    cmd = run.call_args.args[0]
    assert "stream_side_data=rotation" in " ".join(cmd)


def test_probe_video_swaps_dims_on_rotate_tag_270():
    payload = _ffprobe_payload(1280, 720, rotate_tag=270)
    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.subprocess.run"
    ) as run:
        run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
        meta = anim.probe_video("clip.mp4")
    assert meta["width"] == 720
    assert meta["height"] == 1280


def test_stream_anim_decoder_has_fps_filter(tmp_path):
    w, h = 4, 4
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
    encoder = _FakeProc(stderr_bytes=b"")
    pops = []

    def fake_popen(cmd, **kwargs):
        pops.append(list(cmd))
        if "-f" in cmd and cmd[cmd.index("-f") + 1] == "rawvideo" and "pipe:1" in cmd:
            return decoder
        return encoder

    gray = torch.zeros((1, 8, 8), dtype=torch.uint8)
    out_mp4 = tmp_path / "out.mp4"
    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 24.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=gray
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ), patch(
        "core.image.anim._mp4_to_gif"
    ):
        anim.stream_anim("src.mp4", out_mp4, mode="line", cells=2, discord=True, fps=24.0)

    dec_cmd = pops[0]
    assert "-vf" in dec_cmd
    assert dec_cmd[dec_cmd.index("-vf") + 1] == "fps=24"


def test_stream_anim_drains_large_ffmpeg_stderr(tmp_path):
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    huge = b"x" * (1024 * 1024)
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)
    errors = []
    writers = []

    def _pipe_stderr_proc(stdout=None, returncode=0):
        r_fd, w_fd = os.pipe()
        proc = _FakeProc(stdout=stdout, returncode=returncode)
        proc.stderr = os.fdopen(r_fd, "rb", buffering=0)

        def _write():
            try:
                offset = 0
                view = memoryview(huge)
                while offset < len(huge):
                    offset += os.write(w_fd, view[offset : offset + 65536])
            finally:
                os.close(w_fd)

        thread = threading.Thread(target=_write, daemon=True)
        writers.append(thread)
        thread.start()
        return proc

    def fake_popen(cmd, **kwargs):
        if "pipe:1" in cmd:
            return _pipe_stderr_proc(stdout=_FakePipe([frame]))
        return _pipe_stderr_proc()

    def run():
        try:
            anim.stream_anim(
                "src.mp4", tmp_path / "out.mp4", mode="line", cells=1, discord=True
            )
        except Exception as exc:
            errors.append(exc)

    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=gray
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ):
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "render hung on 1MB ffmpeg stderr via os.pipe"
        assert not errors
    for thread in writers:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_stream_anim_gif_via_mp4_to_gif_not_frame_list(tmp_path):
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
    encoder = _FakeProc(stderr_bytes=b"")
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)
    gif_path = tmp_path / "out.gif"
    calls = {"frames_to_gif": 0, "mp4_to_gif": 0}

    def fake_popen(cmd, **kwargs):
        return decoder if "pipe:1" in cmd else encoder

    def fake_mp4_to_gif(mp4, gif):
        calls["mp4_to_gif"] += 1
        Path(gif).write_bytes(b"GIF89a")
        return Path(gif)

    def fake_frames_to_gif(*_a, **_k):
        calls["frames_to_gif"] += 1
        raise AssertionError("must not accumulate frames into GIF")

    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=gray
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ), patch(
        "core.image.anim._mp4_to_gif", side_effect=fake_mp4_to_gif
    ), patch(
        "core.image.anim._frames_to_gif", side_effect=fake_frames_to_gif
    ):
        anim.stream_anim(
            "src.mp4",
            tmp_path / "out.mp4",
            mode="line",
            cells=1,
            discord=True,
            gif_path=gif_path,
        )

    assert calls["mp4_to_gif"] == 1
    assert calls["frames_to_gif"] == 0


def test_even_dims_for_odd_render_size(tmp_path):
    assert anim._even_dims(641, 361) == (642, 362)
    w, h = 4, 4
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
    encoder = _FakeProc(stderr_bytes=b"")
    opened = {}

    def fake_popen(cmd, **kwargs):
        if "pipe:1" in cmd:
            return decoder
        if "-s" in cmd:
            opened["size"] = cmd[cmd.index("-s") + 1]
        return encoder

    odd = torch.zeros((1, 361, 641), dtype=torch.uint8)
    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=odd
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ):
        anim.stream_anim("src.mp4", tmp_path / "out.mp4", mode="line", cells=1, discord=True)

    assert opened["size"] == "642x362"
    ow, oh = map(int, opened["size"].split("x"))
    assert ow % 2 == 0 and oh % 2 == 0


def test_both_render_branches_call_process_rendered_batch(tmp_path):
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)
    real = anim._process_rendered_batch

    def _run(*, discord, mask_pool):
        decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
        encoder = _FakeProc(stderr_bytes=b"")

        def fake_popen(cmd, **kwargs):
            return decoder if "pipe:1" in cmd else encoder

        with ExitStack() as es:
            es.enter_context(patch("core.image.anim.require_tool"))
            es.enter_context(
                patch(
                    "core.image.anim.probe_video",
                    return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
                )
            )
            es.enter_context(
                patch("core.image.anim.pick_device", return_value=torch.device("cpu"))
            )
            es.enter_context(patch("core.image.anim.pick_gpu_batch", return_value=1))
            es.enter_context(patch("core.image.anim._has_nvenc", return_value=False))
            es.enter_context(patch("core.image.anim.render_rgb_batch", return_value=gray))
            es.enter_context(
                patch("core.image.anim.subprocess.Popen", side_effect=fake_popen)
            )
            process_mock = es.enter_context(
                patch("core.image.anim._process_rendered_batch", wraps=real)
            )
            ensure_mock = es.enter_context(
                patch("core.image.anim.ensure_mask_pool", return_value=(object(), 1))
            )
            submit_mock = es.enter_context(
                patch("core.image.anim.submit_line_masks", return_value=([], 0))
            )
            if mask_pool:
                es.enter_context(
                    patch(
                        "core.image.anim.prepare_mask_source",
                        return_value={
                            "rgb": gray.float(),
                            "opts": {},
                            "mask_cpu": np.zeros((1, h, w), dtype=np.float32),
                        },
                    )
                )
                es.enter_context(
                    patch("core.image.anim.collect_line_masks", return_value=None)
                )
                es.enter_context(patch("core.image.anim.close_mask_pool"))
            anim.stream_anim(
                "src.mp4",
                tmp_path / ("pool.mp4" if mask_pool else "direct.mp4"),
                mode="line",
                cells=1,
                discord=discord,
            )
            assert process_mock.call_count >= 1
            if mask_pool:
                assert ensure_mock.called
                assert submit_mock.called
            else:
                assert not ensure_mock.called
                assert not submit_mock.called
            return process_mock.call_count

    assert _run(discord=True, mask_pool=False) >= 1
    assert _run(discord=False, mask_pool=True) >= 1


def test_stream_anim_rotation_affects_out_shape(tmp_path):
    # Stored 8x4 with rot 90 → decoded/display 4x8; out aspect follows swapped meta.
    stored_w, stored_h = 8, 4
    disp_w, disp_h = 4, 8
    frame = np.arange(disp_h * disp_w * 3, dtype=np.uint8).tobytes()
    decoder = _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
    encoder = _FakeProc(stderr_bytes=b"")
    seen = {}

    def fake_render(frames, **kwargs):
        rgb = frames if frames is not None else kwargs.get("rgb_tensor")
        if isinstance(rgb, np.ndarray):
            seen["in_shape"] = rgb.shape
        elif torch.is_tensor(rgb):
            seen["in_shape"] = tuple(rgb.shape)
        # cols=2 → out ~ 16 x 32 with smile 8; keep tiny mock size from aspect
        return torch.zeros((1, 16, 8), dtype=torch.uint8)

    def fake_popen(cmd, **kwargs):
        if "pipe:1" in cmd:
            return decoder
        if "-s" in cmd:
            seen["enc_size"] = cmd[cmd.index("-s") + 1]
        return encoder

    payload = _ffprobe_payload(stored_w, stored_h, frames=1, rotation=90)
    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.subprocess.run"
    ) as run, patch(
        "core.image.anim.pick_device", return_value=torch.device("cpu")
    ), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", side_effect=fake_render
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ):
        run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
        anim.stream_anim(
            "src.mp4", tmp_path / "rot.mp4", mode="line", cells=2, discord=True
        )

    assert seen["in_shape"][-3:-1] == (disp_h, disp_w)
    # Without rotation swap, reshape would be (4, 8) and fail or scramble; enc gets 8x16
    ew, eh = map(int, seen["enc_size"].split("x"))
    assert (ew, eh) == (8, 16)
    assert ew % 2 == 0 and eh % 2 == 0


def test_stderr_logs_unique_sequential_and_parallel(tmp_path, _anim_var_dir):
    seen = []
    lock = threading.Lock()
    real = anim._start_stderr_drain
    w, h = 2, 2
    frame = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    gray = torch.zeros((1, 4, 4), dtype=torch.uint8)

    def tracking(proc, label, render_id):
        thread, chunks, log_path = real(proc, label, render_id)
        with lock:
            seen.append(log_path)
        return thread, chunks, log_path

    def fake_popen(cmd, **kwargs):
        if "pipe:1" in cmd:
            return _FakeProc(stdout=_FakePipe([frame]), stderr_bytes=b"")
        return _FakeProc(stderr_bytes=b"")

    def run_one(name):
        anim.stream_anim(
            "src.mp4",
            tmp_path / f"{name}.mp4",
            mode="line",
            cells=1,
            discord=True,
        )

    with patch("core.image.anim.require_tool"), patch(
        "core.image.anim.probe_video",
        return_value={"width": w, "height": h, "fps": 30.0, "frames": 1},
    ), patch("core.image.anim.pick_device", return_value=torch.device("cpu")), patch(
        "core.image.anim.pick_gpu_batch", return_value=1
    ), patch(
        "core.image.anim._has_nvenc", return_value=False
    ), patch(
        "core.image.anim.render_rgb_batch", return_value=gray
    ), patch(
        "core.image.anim.subprocess.Popen", side_effect=fake_popen
    ), patch(
        "core.image.anim._start_stderr_drain", side_effect=tracking
    ):
        run_one("a")
        run_one("b")
        errors = []

        def _parallel(name):
            try:
                run_one(name)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_parallel, args=("p1",))
        t2 = threading.Thread(target=_parallel, args=("p2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert not t1.is_alive() and not t2.is_alive()
        assert not errors

    assert len(seen) >= 8  # 4 renders × decoder+encoder
    assert len({str(p) for p in seen}) == len(seen)
    for path in seen:
        assert str(os.getpid()) in path.name
        assert path.name.startswith("anim_ffmpeg_")


def test_stderr_log_removed_after_success(tmp_path, _anim_var_dir):
    logs = []
    real = anim._start_stderr_drain

    def tracking(proc, label, render_id):
        thread, chunks, log_path = real(proc, label, render_id)
        logs.append(log_path)
        return thread, chunks, log_path

    with patch("core.image.anim._start_stderr_drain", side_effect=tracking):
        _run_stream_anim(tmp_path, name="ok")

    assert logs
    for path in logs:
        assert not path.exists(), path


def test_stderr_log_kept_on_ffmpeg_error(tmp_path, _anim_var_dir, capsys):
    with pytest.raises(RuntimeError) as exc_info:
        _run_stream_anim(
            tmp_path,
            returncode=1,
            stderr_bytes=b"encoder exploded",
            name="fail",
        )
    message = str(exc_info.value)
    assert "stderr log:" in message
    log_path = Path(message.rsplit("stderr log:", 1)[1].strip())
    assert log_path.exists()
    assert "encoder" in log_path.name or "decoder" in log_path.name
    assert str(anim._VAR_DIR) in str(log_path)
    captured = capsys.readouterr()
    assert str(log_path) in captured.out
