"""core/image/ascii.py, probe.py: ASCII/emoji-арт без GPU и без пула процессов."""
import numpy as np
import pytest
from PIL import Image

from core.image import ascii as art
from core.image.probe import is_animated_media


@pytest.mark.parametrize(
    "raw, mode",
    [("1", "line"), ("line", "line"), ("2", "simple"), ("color", "simple"), ("3", "render"),
     ("pic", "render"), ("4", "details"), ("soft", "details"), (None, "simple")],
)
def test_resolve_mode(raw, mode):
    assert art.resolve_mode(raw) == mode


def test_resolve_mode_unknown():
    with pytest.raises(ValueError):
        art.resolve_mode("hologram")


def test_resolve_size_and_clamp_cells():
    assert art.resolve_size("большой") == ("large", art.DISCORD_LARGE_CELLS)
    assert art.resolve_size(None) == ("small", art.DISCORD_SMALL_CELLS)
    with pytest.raises(ValueError):
        art.resolve_size("huge")
    assert art.clamp_cells(1) == art.MIN_CELLS
    assert art.clamp_cells(10_000) == art.MAX_CELLS
    assert art.clamp_cells(10_000, max_cells=None) == 10_000


def test_resolve_simple_glyphs():
    assert art.resolve_simple_glyphs() == {"dark": art.SIMPLE_DARK_DEFAULT, "light": art.SIMPLE_LIGHT_DEFAULT}
    assert art.resolve_simple_glyphs("#", "") == {"dark": "#", "light": art.SIMPLE_LIGHT_DEFAULT}
    with pytest.raises(ValueError):
        art.resolve_simple_glyphs("ab")
    with pytest.raises(ValueError):
        art.resolve_simple_glyphs("#\n#")
    # Одиночный перевод строки обрезается strip() и даёт глиф по умолчанию.
    assert art.resolve_simple_glyphs("\n")["dark"] == art.SIMPLE_DARK_DEFAULT


def test_render_pic_simple(small_png_bytes):
    text = art.render_pic(small_png_bytes, mode="simple", cells=16)
    assert text.startswith(art.HEADER + "\n")
    rows = art.art_body(text).split("\n")
    assert rows and all(rows)
    glyphs = {art.SIMPLE_DARK_DEFAULT, art.SIMPLE_LIGHT_DEFAULT}
    for row in rows:
        assert set(row) <= set("".join(glyphs))


def test_render_pic_line_without_form_and_parse_back(small_png_bytes):
    text = art.render_pic(small_png_bytes, mode="line", cells=16, form=False)
    assert text.startswith(art.HEADER)
    grid = art.parse_art(text, mode="line")
    rows = art.art_body(text).split("\n")
    assert grid.shape[0] == len(rows) and grid.shape[1] > 0
    assert art.parse_art("").shape == (0, 0)


def test_load_rgb_variants(small_png_bytes):
    gray = np.zeros((4, 4), dtype=np.uint8)
    assert art.load_rgb(gray).shape == (4, 4, 3)
    assert art.load_rgb(Image.new("RGB", (5, 3))).shape[:2] == (3, 5)
    assert art.load_rgb(small_png_bytes).shape[:2] == (64, 64)


def test_fence_and_body():
    text = art.HEADER + "\nab\ncd"
    assert art.art_body(text) == "ab\ncd"
    assert art.fence_pic(text) == art.HEADER + "\n```\nab\ncd\n```"
    assert art.fence_md("xy") == "xy"


def test_discord_chunks_split_by_rows():
    body = "\n".join(f"row{i}" for i in range(70))
    chunks = art.discord_chunks(art.HEADER + "\n" + body)
    assert len(chunks) == 3
    assert chunks[0].startswith(art.HEADER + "\n```")
    assert not chunks[1].startswith(art.HEADER)
    for chunk in chunks:
        assert chunk.endswith("```")
        inner = chunk.split("```")[1].strip("\n").split("\n")
        assert len(inner) <= art.DISCORD_MAX_ROWS


def test_discord_chunks_char_limit():
    body = "\n".join("x" * 50 for _ in range(10))
    chunks = art.discord_chunks(body, limit=130)
    assert len(chunks) > 1
    for chunk in chunks:
        assert art._discord_len(chunk) <= 130


def test_discord_chunks_empty():
    assert art.discord_chunks("") == ["```\n\n```"]


def test_md_to_png(small_png_bytes):
    text = art.render_pic(small_png_bytes, mode="simple", cells=8)
    image = art.md_to_png(text, mode="simple")
    assert image.size[0] > 0 and image.size[1] > 0


def test_is_animated_media(small_png_bytes, animated_gif_bytes):
    assert is_animated_media(b"", "clip.mp4")
    assert not is_animated_media(b"", "a.png")
    assert not is_animated_media(small_png_bytes, "a.png")
    assert is_animated_media(animated_gif_bytes, "a.gif")
