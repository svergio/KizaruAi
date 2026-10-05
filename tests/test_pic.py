import unittest

import numpy as np
from PIL import Image, ImageDraw

from core.image.ascii import (
    COLORS,
    DEFAULT_CELLS,
    DISCORD_LARGE_CELLS,
    DISCORD_MAX_COLS,
    DISCORD_MAX_ROWS,
    DISCORD_SMALL_CELLS,
    CLI_MAX_COLS,
    HEADER,
    LINE_CHARS,
    LINE_PATTERNS,
    MAX_CELLS,
    RAMP,
    STICKERS,
    _SHADE_COUNT,
    discord_chunks,
    fence_md,
    fence_pic,
    line_indexes,
    md_to_png,
    render_png,
    load_rgb,
    pick_blocks,
    quads_from_patch,
    quadtree_form,
    render_pic,
    resolve_mode,
    resolve_size,
    simple_indexes,
)


def _solid(color, size=64):
    array = np.zeros((size, size, 3), dtype=np.float32)
    array[:, :] = color
    return array


class PicTests(unittest.TestCase):
    def test_dictionary_is_eleven_gray_steps(self):
        chars = [item[0] for item in STICKERS]
        self.assertEqual(len(chars), 11)
        self.assertEqual(len(RAMP), 11)
        self.assertEqual(len(chars), len(set(chars)))
        self.assertEqual(len(COLORS), 11)
        levels = [float(color.mean()) for color in COLORS]
        self.assertEqual(levels, sorted(levels))
        self.assertLess(levels[0], 20)
        self.assertGreater(levels[-1], 240)

    def test_line_dictionary_covers_quadrants(self):
        chars = list(LINE_CHARS)
        self.assertEqual(len(chars), len(set(chars)))
        for needed in ("▀", "▖", "▙", "▚", "▛", "█", "▓", " "):
            self.assertIn(needed, chars)
        self.assertEqual(LINE_PATTERNS.shape, (len(chars), 4))
        shade_levels = [float(LINE_PATTERNS[i].mean()) for i in range(_SHADE_COUNT)]
        self.assertGreaterEqual(len(shade_levels), 8)
        self.assertEqual(shade_levels, sorted(shade_levels))
        from core.image.ascii import _MEASURED_COUNT

        self.assertGreaterEqual(_MEASURED_COUNT, 50)
        rounded = np.round(LINE_PATTERNS[_SHADE_COUNT:], 1)
        unique = {tuple(row) for row in rounded}
        self.assertGreaterEqual(len(unique), 16)
        patterns = {char: np.array(pattern, dtype=np.float32) for char, pattern in zip(LINE_CHARS, LINE_PATTERNS)}
        anchor = patterns["▛"]
        self.assertLess(float(np.linalg.norm(anchor - np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32))), 0.21)
        self.assertIn("▀", patterns)
        self.assertEqual(tuple(patterns["█"]), (1.0, 1.0, 1.0, 1.0))

    def test_transparent_pixels_composite_to_black(self):
        sticker = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
        ImageDraw.Draw(sticker).rectangle((4, 4, 11, 11), fill=(200, 120, 80, 255))
        rgb = load_rgb(sticker)
        self.assertTrue(np.all(rgb[0, 0] == 0))
        self.assertGreater(float(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2].mean()), 50)

    def test_transparent_margins_are_cropped(self):
        sticker = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        ImageDraw.Draw(sticker).rectangle((16, 20, 47, 43), fill=(200, 120, 80, 255))
        rgb = load_rgb(sticker)
        self.assertLessEqual(rgb.shape[0], 28)
        self.assertLessEqual(rgb.shape[1], 36)
        self.assertGreater(float(rgb.mean()), 20)

    def test_flat_keeps_shade_and_split_keeps_line(self):
        from core.image.ascii import _MEASURED_COUNT

        flat = np.array([[[0.25, 0.25, 0.28, 0.22]]], dtype=np.float32)
        shade_chars = set(LINE_CHARS[:_SHADE_COUNT])
        measured_chars = set(LINE_CHARS[_SHADE_COUNT:_SHADE_COUNT + _MEASURED_COUNT])
        picked = LINE_CHARS[int(pick_blocks(flat)[0, 0])]
        self.assertIn(picked, shade_chars)
        self.assertNotIn(picked, measured_chars)
        split = np.array([[[1.0, 1.0, 1.0, 0.0]]], dtype=np.float32)
        self.assertEqual(LINE_CHARS[int(pick_blocks(split)[0, 0])], "▛")
        soft = np.array([[[1.0, 1.0, 0.5, 0.0]]], dtype=np.float32)
        soft_idx = int(pick_blocks(soft)[0, 0])
        soft_char = LINE_CHARS[soft_idx]
        self.assertIn(soft_char, measured_chars)
        target = np.array([1.0, 1.0, 0.5, 0.0], dtype=np.float32)
        diff = LINE_PATTERNS[:, :] - target
        nearest = int(np.argmin(np.sum(diff * diff, axis=-1)))
        self.assertEqual(soft_idx, nearest)
        self.assertLess(float(np.linalg.norm(LINE_PATTERNS[soft_idx] - target)), 0.55)

    def test_simple_matches_average_tone(self):
        black = simple_indexes(_solid((0, 0, 0)), cells=8)
        white = simple_indexes(_solid((255, 255, 255)), cells=8)
        mid = simple_indexes(_solid((128, 128, 128)), cells=8)
        self.assertLess(float(COLORS[black[0, 0]].mean()), 20)
        self.assertGreater(float(COLORS[white[0, 0]].mean()), 240)
        self.assertGreater(float(COLORS[mid[0, 0]].mean()), float(COLORS[black[0, 0]].mean()))
        self.assertLess(float(COLORS[mid[0, 0]].mean()), float(COLORS[white[0, 0]].mean()))

    def test_simple_uses_four_by_four(self):
        rgb = _solid((0, 0, 0), size=32)
        rgb[:, 16:] = (255, 255, 255)
        indexes = simple_indexes(rgb, cells=8)
        self.assertEqual(indexes.shape, (8, 8))
        self.assertLess(float(COLORS[indexes[0, 0]].mean()), float(COLORS[indexes[0, -1]].mean()))

    def test_aspect_follows_the_picture(self):
        tile = _solid((40, 40, 40), size=80)
        wide = np.concatenate([tile, tile], axis=1)
        self.assertEqual(wide.shape, (80, 160, 3))
        indexes = simple_indexes(wide, cells=16)
        self.assertEqual(indexes.shape, (8, 16))
        tall = np.concatenate([tile, tile], axis=0)
        self.assertEqual(tall.shape, (160, 80, 3))
        tall_indexes = simple_indexes(tall, cells=16)
        self.assertEqual(tall_indexes.shape, (32, 16))
        square = simple_indexes(_solid((40, 40, 40), size=64), cells=12)
        self.assertEqual(square.shape, (12, 12))

    def test_patch_downsamples_to_four_pixels(self):
        block = np.zeros((16, 16, 3), dtype=np.float32)
        block[:, 8:] = (255, 255, 255)
        quads = quads_from_patch(block)
        self.assertEqual(quads.shape, (4, 3))
        self.assertLess(float(quads[0].mean()), 20)
        self.assertGreater(float(quads[1].mean()), 230)
        self.assertLess(float(quads[2].mean()), 20)
        self.assertGreater(float(quads[3].mean()), 230)

    def test_quadtree_keeps_line_and_skips_flat(self):
        flat = np.full((64, 64), 0.4, dtype=np.float32)
        self.assertEqual(float(quadtree_form(flat, 8, 8).max()), 0.0)
        lined = flat.copy()
        lined[:, 32:34] = 0.0
        form = quadtree_form(lined, 8, 8)
        self.assertGreater(float(form[:, 4].max()), 0.0)
        self.assertEqual(float(form[:, 0].max()), 0.0)

    def test_line_overwrites_flat_shade(self):
        from core.image.ascii import _MEASURED_COUNT

        image = Image.new("RGB", (128, 128), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.line((64, 0, 64, 127), fill=(0, 0, 0), width=3)
        rgb = np.asarray(image, dtype=np.float32)
        chars = LINE_CHARS[line_indexes(rgb, cells=16, patch=16)]
        shade_chars = set(LINE_CHARS[:_SHADE_COUNT])
        measured_chars = set(LINE_CHARS[_SHADE_COUNT:_SHADE_COUNT + _MEASURED_COUNT])
        corner_blocks = {"▘", "▝", "▖", "▗", "▚", "▞", "▜", "▙", "▟"}
        self.assertTrue(set(chars[:, 1]).isdisjoint(corner_blocks))
        self.assertTrue(any(char in measured_chars for char in chars[:, 7]))
        self.assertTrue(any(char not in shade_chars for char in chars[:, 7]))

    def test_line_survives_downscale(self):
        image = Image.new("RGB", (128, 128), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.line((64, 0, 64, 127), fill=(5, 5, 5), width=2)
        rgb = np.asarray(image, dtype=np.float32)
        indexes = line_indexes(rgb, cells=16, patch=16)
        self.assertEqual(indexes.shape, (16, 16))
        light = LINE_PATTERNS[indexes].mean(axis=-1)
        self.assertLess(float(light[:, 8].mean()), float(light[:, 1].mean()))
        text = render_pic(rgb, mode="1", cells=16)
        self.assertTrue(text.startswith(HEADER))
        body = text.split("\n", 1)[1].split("\n")
        self.assertEqual(len(body), 16)

    def test_modes(self):
        self.assertEqual(resolve_mode("1"), "line")
        self.assertEqual(resolve_mode("line"), "line")
        self.assertEqual(resolve_mode("2"), "simple")
        self.assertEqual(resolve_mode("simple"), "simple")

    def test_fence_pic_wraps_art(self):
        image = Image.new("RGB", (32, 32), (20, 20, 20))
        text = render_pic(image, mode="simple", cells=8)
        fenced = fence_pic(text)
        self.assertTrue(fenced.startswith(HEADER + "\n```\n"))
        self.assertTrue(fenced.endswith("\n```"))
        body = text.split("\n", 1)[1]
        self.assertIn(body, fenced)
        md = fence_md(text)
        self.assertFalse(md.startswith(HEADER))
        self.assertNotIn("```", md)
        self.assertEqual(md, body)

    def test_render_png_matches_grid(self):
        from core.image.ascii import OUTPUT_SMILE_PX

        image = Image.new("RGB", (64, 32), (20, 20, 20))
        png = render_png(image, mode="line", cells=16, discord=True, scale=2)
        self.assertEqual(png.size, (16 * OUTPUT_SMILE_PX, 8 * OUTPUT_SMILE_PX))

    def test_md_to_png_from_symbols(self):
        from core.image.ascii import LINE_CHARS, OUTPUT_SMILE_PX, glyph_atlas

        text = render_pic(Image.new("RGB", (64, 32), (20, 20, 20)), mode="line", cells=16)
        md = fence_md(text)
        png = md_to_png(md, mode="line")
        self.assertEqual(png.size, (16 * OUTPUT_SMILE_PX, 8 * OUTPUT_SMILE_PX))
        atlas = glyph_atlas("line")
        self.assertEqual(atlas.shape, (len(LINE_CHARS), OUTPUT_SMILE_PX, OUTPUT_SMILE_PX))
        block = atlas[list(LINE_CHARS).index("▛")]
        self.assertGreater(float(block.max()), 0)
        self.assertGreater(float(block.std()), 5)
        self.assertEqual(OUTPUT_SMILE_PX * 320, 2560)

    def test_discord_chunks_use_markdown_fence(self):
        image = Image.new("RGB", (48, 48), (20, 20, 20))
        text = render_pic(image, mode="simple", cells=20)
        chunks = discord_chunks(text, limit=180)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith(HEADER + "\n```\n"))
        self.assertTrue(chunks[0].endswith("\n```"))
        self.assertTrue(chunks[1].startswith("```\n"))
        self.assertTrue(chunks[1].endswith("\n```"))
        self.assertLessEqual(max(len(chunk.encode("utf-16-le")) // 2 for chunk in chunks), 180)

    def test_discord_fence_budget_constants(self):
        self.assertEqual(DISCORD_MAX_ROWS, 31)
        self.assertEqual(DISCORD_MAX_COLS, 102)
        self.assertEqual(DEFAULT_CELLS, DISCORD_SMALL_CELLS)
        self.assertEqual(MAX_CELLS, DISCORD_LARGE_CELLS)
        self.assertEqual(resolve_size("маленький"), ("small", 31))
        self.assertEqual(resolve_size("большой"), ("large", 102))
        image = Image.new("RGB", (256, 256), (20, 20, 20))
        text = render_pic(image, mode="line", discord=True)
        body = text.split("\n", 1)[1].split("\n")
        self.assertEqual(len(body), DEFAULT_CELLS)
        self.assertEqual(len(body[0]), DEFAULT_CELLS)
        chunks = discord_chunks(text)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].startswith(HEADER + "\n```\n"))
        self.assertTrue(chunks[0].endswith("\n```"))
        large = render_pic(image, mode="line", cells=DISCORD_LARGE_CELLS, discord=True)
        large_body = large.split("\n", 1)[1].split("\n")
        self.assertEqual(len(large_body[0]), DISCORD_MAX_COLS)
        tall = render_pic(Image.new("RGB", (64, 256), (20, 20, 20)), mode="line", cells=102)
        tall_rows = tall.split("\n", 1)[1].split("\n")
        self.assertGreater(len(tall_rows), DISCORD_MAX_ROWS)
        tall_chunks = discord_chunks(tall)
        self.assertGreater(len(tall_chunks), 1)
        for chunk in tall_chunks:
            self.assertTrue(chunk.endswith("\n```"))
            inner = chunk.split("```\n", 1)[1].rsplit("\n```", 1)[0]
            self.assertLessEqual(len(inner.split("\n")), DISCORD_MAX_ROWS)

    def test_cli_export_uses_md_cap(self):
        self.assertEqual(CLI_MAX_COLS, 320)
        image = Image.new("RGB", (256, 256), (20, 20, 20))
        text = render_pic(image, mode="line", cells=CLI_MAX_COLS, discord=False)
        body = text.split("\n", 1)[1].split("\n")
        self.assertEqual(len(body), CLI_MAX_COLS)
        self.assertEqual(len(body[0]), CLI_MAX_COLS)
        capped = render_pic(image, mode="line", cells=400, discord=False)
        capped_body = capped.split("\n", 1)[1].split("\n")
        self.assertEqual(len(capped_body[0]), CLI_MAX_COLS)

    def test_small_source_upscales_to_full_grid(self):
        from core.image.ascii import _grid, _prepare_for_grid

        tiny = np.zeros((20, 20, 3), dtype=np.float32)
        tiny[:, 8:12] = 255.0
        for cells, discord in ((31, True), (102, True), (CLI_MAX_COLS, False)):
            cols, rows = _grid(20, 20, cells, discord=discord)
            prepared = _prepare_for_grid(tiny, cols, rows)
            self.assertEqual(prepared.shape[:2], (rows * 2, cols * 2))
            text = render_pic(tiny, mode="line", cells=cells, discord=discord)
            body = text.split("\n", 1)[1].split("\n")
            self.assertEqual(len(body), rows)
            self.assertEqual(len(body[0]), cols)

    def test_bot_has_pic_slash(self):
        import tempfile
        from pathlib import Path

        from app.discord.__main__ import build_bot
        with tempfile.TemporaryDirectory() as folder:
            bot = build_bot(db_path=Path(folder) / "bot.db")
            group = bot.tree.get_command("ai")
            self.assertIn("pic", {command.name for command in group.commands})

    def test_downsample_quads_vectorized_shape(self):
        from core.image.ascii import downsample_quads

        field = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(48, 64)
        quads = downsample_quads(field, cols=16, rows=12, reduce="mean")
        self.assertEqual(quads.shape, (12, 16, 4))
        self.assertTrue(np.all(quads >= 0.0))
        self.assertTrue(np.all(quads <= 1.0))
        max_quads = downsample_quads(field, cols=8, rows=6, reduce="max")
        self.assertEqual(max_quads.shape, (6, 8, 4))

    def test_line_indexes_form_flag(self):
        rgb = _solid((40, 40, 40), size=64)
        with_form = line_indexes(rgb, cells=16, discord=True, form=True)
        no_form = line_indexes(rgb, cells=16, discord=True, form=False)
        self.assertEqual(with_form.shape, no_form.shape)
        self.assertEqual(no_form.shape, (16, 16))

    def test_render_fast_frame_smoke(self):
        from core.image.render import pick_device, render_frame_array

        image = Image.new("RGB", (128, 72), (20, 20, 20))
        array = render_frame_array(image, mode="line", cells=32, discord=False, scale=2)
        # rows = round(32 * 72/128) = 18; смайл 8px
        self.assertEqual(array.shape, (18 * 8, 32 * 8))
        self.assertEqual(array.dtype, np.uint8)
        self.assertIn(pick_device().type, {"cpu", "cuda"})

    def test_cdist_matches_squared_distance(self):
        import torch

        from core.image.render import _pattern_indexes_torch, _patterns_tensor

        torch.manual_seed(0)
        quads = torch.rand(4, 5, 4)
        patterns = _patterns_tensor(torch.device("cpu"))
        got = _pattern_indexes_torch(quads, patterns)
        diff = quads[:, :, None, :] - patterns.view(1, 1, -1, 4)
        expect = torch.argmin((diff * diff).sum(dim=-1), dim=-1)
        self.assertTrue(torch.equal(got, expect))

    def test_anim_uses_default_cells_not_width_over_4(self):
        import torch

        from core.image.anim import cells_for_source
        from core.image.ascii import CLI_DEFAULT_CELLS, OUTPUT_SMILE_PX
        from core.image.render import render_rgb_batch

        self.assertEqual(cells_for_source(640), CLI_DEFAULT_CELLS)
        self.assertEqual(CLI_DEFAULT_CELLS, 320)
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        device = torch.device("cpu")
        out_s2 = render_rgb_batch(
            frame,
            mode="line",
            cells=32,
            scale=2,
            discord=False,
            device=device,
            cap=False,
            upscale=True,
        )
        out_s16 = render_rgb_batch(
            frame,
            mode="line",
            cells=32,
            scale=16,
            discord=False,
            device=device,
            cap=False,
            upscale=True,
        )
        self.assertEqual(out_s2.shape[-1], 32 * OUTPUT_SMILE_PX)
        self.assertEqual(out_s16.shape[-1], out_s2.shape[-1])
        self.assertEqual(out_s16.shape[-2], out_s2.shape[-2])

    def test_simple_png_width_is_cells_times_8(self):
        from core.image.ascii import OUTPUT_SMILE_PX

        image = Image.new("RGB", (48, 24), (30, 30, 30))
        png = render_png(image, mode="simple", cells=32, discord=False, scale=2, upscale=True)
        self.assertEqual(png.size[0], 32 * OUTPUT_SMILE_PX)

    def test_simple_glyph_tiles_have_ink_on_ramp(self):
        from core.image.ascii import OUTPUT_SMILE_PX, SIMPLE_CHARS, glyph_atlas

        atlas = glyph_atlas("simple")
        self.assertEqual(atlas.shape, (len(SIMPLE_CHARS), OUTPUT_SMILE_PX, OUTPUT_SMILE_PX))
        for idx in range(len(atlas)):
            self.assertGreater(float(atlas[idx].max()), 0)
        for idx in (0, 5, 6, 10):
            block = atlas[idx]
            self.assertGreater(float(block.std()), 5)

    def test_upscale_png_width_is_cells_times_8(self):
        from core.image.ascii import OUTPUT_SMILE_PX

        image = Image.new("RGB", (48, 24), (30, 30, 30))
        sizes = []
        for scale in (1, 16):
            png = render_png(image, mode="line", cells=32, discord=False, scale=scale, upscale=True)
            self.assertEqual(png.size[0], 32 * OUTPUT_SMILE_PX)
            sizes.append(png.size)
        self.assertEqual(sizes[0], sizes[1])
        wide = render_png(
            Image.new("RGB", (20, 8), (12, 12, 12)),
            mode="line",
            cells=320,
            discord=False,
            scale=1,
            upscale=True,
        )
        self.assertEqual(wide.size[0], 2560)

    def test_no_upscale_width_2000_scale_16_is_62_cols(self):
        from core.image.ascii import OUTPUT_SMILE_PX, resolve_frame

        cols, _rows = resolve_frame(2000, 100, 320, 16, upscale=False, discord=False)
        self.assertEqual(cols, 62)
        rgb = np.zeros((100, 2000, 3), dtype=np.float32)
        png = render_png(rgb, mode="line", cells=320, discord=False, scale=16, upscale=False)
        self.assertEqual(png.size[0], 496)
        self.assertEqual(png.size[0], cols * OUTPUT_SMILE_PX)

    def test_quadrant_block_matches_scale_and_line_replaces_shade(self):
        from core.image.ascii import compose_line_indexes, shade_indexes, smile_blocks

        field2 = np.arange(16, dtype=np.float32).reshape(4, 4)
        quads2, _means2 = smile_blocks(field2, rows=1, cols=1, scale=2)
        self.assertEqual(quads2.shape, (1, 1, 4))
        self.assertAlmostEqual(float(quads2[0, 0, 0]), float(field2[0:2, 0:2].mean()))
        self.assertAlmostEqual(float(quads2[0, 0, 1]), float(field2[0:2, 2:4].mean()))
        self.assertAlmostEqual(float(quads2[0, 0, 2]), float(field2[2:4, 0:2].mean()))
        self.assertAlmostEqual(float(quads2[0, 0, 3]), float(field2[2:4, 2:4].mean()))

        field4 = np.arange(64, dtype=np.float32).reshape(8, 8)
        quads4, _means4 = smile_blocks(field4, rows=1, cols=1, scale=4)
        self.assertAlmostEqual(float(quads4[0, 0, 0]), float(field4[0:4, 0:4].mean()))
        self.assertNotAlmostEqual(float(quads4[0, 0, 0]), float(field4[0:2, 0:2].mean()), places=3)

        quads = np.array([[[1.0, 1.0, 1.0, 0.0]]], dtype=np.float32)
        means = np.array([[0.75]], dtype=np.float32)
        indexes = compose_line_indexes(quads, means)
        background = int(shade_indexes(means)[0, 0])
        chosen = int(indexes[0, 0])
        self.assertGreaterEqual(chosen, _SHADE_COUNT)
        self.assertNotEqual(chosen, background)
        self.assertEqual(LINE_CHARS[chosen], "▛")
        glyph = LINE_PATTERNS[chosen]
        self.assertGreater(float(glyph.std()), 0.1)
        self.assertFalse(np.allclose(glyph, np.full(4, float(means[0, 0]))))

    def test_rgb_batch_shape_matches_single_frame(self):
        import torch

        from core.image.render import render_frame_array, render_rgb_batch

        image = Image.new("RGB", (128, 72), (30, 30, 30))
        device = torch.device("cpu")
        single = render_frame_array(image, mode="line", cells=32, discord=False, scale=2, device=device)
        rgb = np.asarray(image)
        batch = render_rgb_batch(
            np.stack([rgb, rgb]),
            mode="line",
            cells=32,
            scale=2,
            discord=False,
            device=device,
            cap=True,
        )
        self.assertEqual(batch.shape[0], 2)
        self.assertEqual(batch.shape[1:], single.shape)

    def test_cli_line_mask_comes_from_original_not_scale(self):
        from core.image.ascii import line_mask

        flat = np.full((96, 96, 3), 140, dtype=np.float32)
        flat_idx = line_indexes(flat, cells=16, discord=False, scale=8, upscale=True)
        self.assertTrue(np.all(flat_idx < _SHADE_COUNT))
        self.assertEqual(float(line_mask(np.full((96, 96), 0.55, dtype=np.float32)).max()), 0.0)

        rgb = np.full((256, 512, 3), 210, dtype=np.float32)
        rgb[:, 128:130] = 0
        mask = line_mask(np.asarray(rgb[:, :, 0] / 255.0, dtype=np.float32))
        self.assertEqual(mask.shape, (256, 512))
        self.assertEqual(float(mask[:, :96].max()), 0.0)
        self.assertGreater(float(mask[:, 120:140].max()), 0.0)

        indexes = line_indexes(rgb, cells=16, discord=False, scale=16, upscale=True)
        self.assertTrue(np.any(indexes[:, 4] >= _SHADE_COUNT))
        self.assertTrue(np.all(indexes[:, 0] < _SHADE_COUNT))
        self.assertTrue(np.all(indexes[:, -1] < _SHADE_COUNT))

    def test_line_ink_follows_neighbor_brightness(self):
        from core.image.ascii import tint_line_quads

        photo = np.array([[[[0.8, 0.8, 0.8, 0.8]]]], dtype=np.float32)
        mask = np.array([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=np.float32)
        hard = tint_line_quads(photo, mask, contrast=1.0)
        soft = tint_line_quads(photo, mask, contrast=0.5)
        self.assertAlmostEqual(float(hard[0, 0, 0, 1]), 0.0, places=5)
        self.assertGreater(float(soft[0, 0, 0, 1]), 0.3)
        self.assertLess(float(soft[0, 0, 0, 1]), 0.5)
        self.assertAlmostEqual(float(soft[0, 0, 0, 0]), 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
