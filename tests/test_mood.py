import asyncio
import tempfile
import unittest
from pathlib import Path

from core.mood.model import (
    fit_mood,
    load_mood,
    mood_source,
    predict_mood,
    save_mood,
)
from app.discord.__main__ import replied_content


class MoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vocab = fit_mood(epochs=300, seed=2)

    def _assert_axes(self, text, exp_kind, exp_fun):
        kind_name, kind, fun_name, fun = predict_mood(self.model, self.vocab, text)
        self.assertEqual(kind_name, exp_kind, (text, kind, fun))
        self.assertEqual(fun_name, exp_fun, (text, kind, fun))

    def test_axis_phrases(self):
        self._assert_axes("злой и грустный", "злой", "грустный")
        self._assert_axes("добрый но грустный", "добрый", "грустный")
        self._assert_axes("злой но веселый", "злой", "веселый")
        self._assert_axes("круто", "добрый", "веселый")

    def test_reply_beats_inline_text(self):
        self.assertEqual(mood_source("мне хуево", "круто"), "мне хуево")
        self.assertEqual(mood_source("  ", "круто"), "круто")
        self.assertEqual(mood_source("", ""), "")

    def test_mood_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mood.pt"
            save_mood(path, self.model, self.vocab, train_phrase_acc=1.0)
            loaded, vocab, meta = load_mood(path)
        self.assertEqual(vocab, self.vocab)
        self.assertEqual(meta["train_phrase_acc"], 1.0)
        kind_name, _, fun_name, _ = predict_mood(loaded, vocab, "злой и грустный")
        self.assertEqual(kind_name, "злой")
        self.assertEqual(fun_name, "грустный")

    def test_reject_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            save_mood(path, self.model, self.vocab)
            payload = __import__("torch").load(path, weights_only=False)
            del payload["kind"]
            __import__("torch").save(payload, path)
            with self.assertRaises(ValueError):
                load_mood(path)

    def test_replied_message_content(self):
        class Message:
            content = "хуево"

        class Reference:
            message_id = 7
            resolved = Message()

        class CommandMessage:
            reference = Reference()

        class Ctx:
            message = CommandMessage()

        text = asyncio.run(replied_content(Ctx()))
        self.assertEqual(text, "хуево")
        kind_name, _, fun_name, _ = predict_mood(self.model, self.vocab, text)
        self.assertEqual(kind_name, "злой")
        self.assertEqual(fun_name, "грустный")


if __name__ == "__main__":
    unittest.main()
