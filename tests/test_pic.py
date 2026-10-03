import unittest

import numpy as np
from PIL import Image

from kizaru.pic import (
    COLORS,
    HEADER,
    STICKERS,
    discord_chunks,
    line_indexes,
    render_pic,
    resolve_mode,
    simple_indexes,
)


def _solid(color, size=64):
    array = np.zeros((size, size, 3), dtype=np.float32)
    array[:, :] = color
    return array


class PicTests(unittest.TestCase):
    def test_dictionary_is_large(self):
        chars = [item[0] for item in STICKERS]
        self.assertGreaterEqual(len(chars), 80)
        self.assertEqual(len(chars), len(set(chars)))
        self.assertEqual(len(COLORS), sum(1 for item in STICKERS if item[3] == "flat"))

    def test_simple_matches_average_color(self):
        black = simple_indexes(_solid((0, 0, 0)), cells=8)
        white = simple_indexes(_solid((255, 255, 255)), cells=8)
        red = simple_indexes(_solid((220, 20, 30)), cells=8)
        self.assertLess(float(COLORS[black[0, 0]].mean()), 40)
        self.assertGreater(float(COLORS[white[0, 0]].mean()), 220)
        self.assertGreater(float(COLORS[red[0, 0], 0]), float(COLORS[red[0, 0], 1]))
        self.assertGreater(float(COLORS[red[0, 0], 0]), float(COLORS[red[0, 0], 2]))

    def test_simple_uses_four_by_four(self):
        rgb = _solid((0, 0, 0), size=32)
        rgb[:, 16:] = (255, 255, 255)
        indexes = simple_indexes(rgb, cells=8)
        self.assertEqual(indexes.shape, (8, 8))
        self.assertLess(float(COLORS[indexes[0, 0]].mean()), float(COLORS[indexes[0, -1]].mean()))

    def test_line_mode_is_square_and_keeps_edges(self):
        rgb = _solid((250, 250, 250), size=80)
        rgb[:, :40] = (5, 5, 5)
        indexes = line_indexes(rgb, cells=16)
        self.assertEqual(indexes.shape, (16, 16))
        text = render_pic(rgb, mode="1", cells=16)
        self.assertTrue(text.startswith(HEADER))
        self.assertGreater(len(text.split("\n")), 10)

    def test_modes(self):
        self.assertEqual(resolve_mode("1"), "line")
        self.assertEqual(resolve_mode("line"), "line")
        self.assertEqual(resolve_mode("2"), "simple")
        self.assertEqual(resolve_mode("simple"), "simple")

    def test_discord_chunks_keep_rows(self):
        image = Image.new("RGB", (48, 48), (20, 20, 20))
        text = render_pic(image, mode="simple", cells=20)
        chunks = discord_chunks(text, limit=180)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith(HEADER))
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 180)

    def test_bot_has_pic_command(self):
        from bot import build_bot
        from kizaru.model import TextRNN

        model = TextRNN(4, 8, 8, n_layers=1)
        bot = build_bot(model, {"a": 0}, {0: "a"}, "cpu")
        self.assertIn("pic", {command.name for command in bot.commands})


if __name__ == "__main__":
    unittest.main()
