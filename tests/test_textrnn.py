import tempfile
import unittest
from pathlib import Path

import torch

from core.text.checkpoint import load_model, save_model
from core.corpus.lyrics import get_batch, load_corpus, text_to_seq
from core.text.rnn import TextRNN
from core.text.sample import _completed_lyric_lines, evaluate


SAMPLE = (
    "Я живу на районе, cousin, это мой дом.\n"
    "Ночью город не спит, и я снова за рулём.\n"
    "Hey, keep it real, не забудь про свой путь.\n"
) * 40


class TextRNNTests(unittest.TestCase):
    def test_completed_lyric_lines_ignores_blanks(self):
        text = "куплет раз\n\nкуплет два\n\nкуплет три\n"
        self.assertEqual(_completed_lyric_lines(text), 3)
        self.assertEqual(text.count("\n"), 5)
        self.assertEqual(_completed_lyric_lines("Я без перевода"), 0)

    def test_max_lines_skips_blank_newlines(self):
        local = SAMPLE + "!"
        _, char_to_idx, idx_to_char = text_to_seq(local)
        model = TextRNN(
            input_size=len(idx_to_char),
            hidden_size=16,
            embedding_size=16,
            n_layers=1,
        )
        # Fill / blank / fill / blank / fill, then junk that must not appear after stop.
        script = list("дом.\n\nпуть.\n\nгород\n!!!!")
        ids = [char_to_idx[ch] for ch in script]
        cursor = {"i": 0}

        def fake_choice(_n, p=None):
            i = cursor["i"]
            cursor["i"] = min(i + 1, len(ids) - 1)
            return ids[i]

        import numpy as np

        original = np.random.choice
        np.random.choice = fake_choice
        try:
            lined = evaluate(
                model,
                char_to_idx,
                idx_to_char,
                start_text="Я",
                prediction_len=80,
                temp=0.8,
                max_lines=3,
            )
        finally:
            np.random.choice = original

        # rstrip() drops the final newline, so count visible non-empty lines.
        non_empty = sum(1 for line in lined.splitlines() if line.strip())
        self.assertEqual(non_empty, 3)
        self.assertGreater(lined.count("\n"), 3)
        self.assertNotIn("!", lined)

    def test_text_to_seq_roundtrip(self):
        sequence, char_to_idx, idx_to_char = text_to_seq(SAMPLE)
        restored = "".join(idx_to_char[int(idx)] for idx in sequence)
        self.assertEqual(restored, SAMPLE)
        self.assertEqual(len(char_to_idx), len(idx_to_char))

    def test_get_batch_shapes(self):
        sequence, _, _ = text_to_seq(SAMPLE)
        train, target = get_batch(sequence, seq_len=32, batch_size=4)
        self.assertEqual(tuple(train.shape), (4, 31, 1))
        self.assertEqual(tuple(target.shape), (4, 31, 1))
        self.assertTrue(torch.equal(train[:, 1:, 0], target[:, :-1, 0]))

    def test_forward_and_generate(self):
        sequence, char_to_idx, idx_to_char = text_to_seq(SAMPLE)
        model = TextRNN(
            input_size=len(idx_to_char),
            hidden_size=32,
            embedding_size=32,
            n_layers=2,
        )
        train, _ = get_batch(sequence, seq_len=32, batch_size=4)
        train = train.permute(1, 0, 2)
        output, hidden_out = model(train, model.init_hidden(4))
        self.assertEqual(tuple(output.shape), (31, 4, len(idx_to_char)))
        self.assertEqual(tuple(hidden_out[0].shape), (2, 4, 32))

        logits, stepped = model.step(torch.LongTensor([char_to_idx["Я"]]), model.init_hidden(1))
        self.assertEqual(tuple(logits.shape), (1, 1, len(idx_to_char)))
        self.assertEqual(tuple(stepped[0].shape), (2, 1, 32))

        text = evaluate(model, char_to_idx, idx_to_char, start_text="Я", prediction_len=20, temp=0.8)
        self.assertTrue(text.startswith("Я"))
        self.assertGreaterEqual(len(text), 15)

        lined = evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text="Я",
            prediction_len=80,
            temp=0.8,
            max_lines=2,
        )
        self.assertLessEqual(
            _completed_lyric_lines(lined) - _completed_lyric_lines("Я"), 2
        )

    def test_export_file_roundtrip(self):
        _, char_to_idx, idx_to_char = text_to_seq(SAMPLE)
        model = TextRNN(len(idx_to_char), 16, 16, n_layers=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kizaru.pt"
            save_model(path, model, char_to_idx, idx_to_char, epoch=3, mean_loss=1.2)
            loaded, loaded_chars, loaded_idx, meta = load_model(path, device=torch.device("cpu"))
        self.assertEqual(loaded_chars, char_to_idx)
        self.assertEqual(meta["epoch"], 3)
        self.assertAlmostEqual(meta["mean_loss"], 1.2)
        text = evaluate(loaded, loaded_chars, loaded_idx, start_text="Я", prediction_len=12, temp=0.5)
        self.assertTrue(text.startswith("Я"))

        bare = TextRNN(len(idx_to_char), 16, 16, n_layers=1)
        bare.char_to_idx = char_to_idx
        bare.idx_to_char = idx_to_char
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bare.pt"
            torch.save(bare, path)
            loaded_bare, _, _, _ = load_model(path, device=torch.device("cpu"))
        self.assertIsInstance(loaded_bare, TextRNN)

    def test_fallback_messages_loadable(self):
        path = Path("data/corpus/messages_fallback.csv")
        self.assertTrue(path.exists())
        self.assertGreater(len(load_corpus(path)), 1000)

    def test_bot_ai_slash_group(self):
        from app.discord.__main__ import build_bot

        with tempfile.TemporaryDirectory() as folder:
            bot = build_bot(db_path=Path(folder) / "bot.db")
            group = bot.tree.get_command("ai")
            self.assertIsNotNone(group)
            names = {command.name for command in group.commands}
            self.assertEqual(names, {"text", "mood", "pic", "help"})
            self.assertEqual(len(list(bot.commands)), 0)


if __name__ == "__main__":
    unittest.main()
