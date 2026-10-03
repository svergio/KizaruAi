import asyncio
import tempfile
import unittest
from pathlib import Path

from kizaru.mood import (
    HOLDOUT,
    NOTEBOOK_TEST,
    accuracy,
    fit_mood,
    load_mood,
    mood_source,
    predict_mood,
    save_mood,
)
from bot import replied_content


class MoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vocab = fit_mood(epochs=300, seed=1)

    def test_notebook_holdout(self):
        score = accuracy(self.model, self.vocab, NOTEBOOK_TEST)
        self.assertEqual(score, 1.0)

    def test_unseen_phrases(self):
        score = accuracy(self.model, self.vocab, HOLDOUT)
        self.assertEqual(score, 1.0)

    def test_reply_beats_inline_text(self):
        self.assertEqual(mood_source("мне хуево", "круто"), "мне хуево")
        self.assertEqual(mood_source("  ", "круто"), "круто")
        self.assertEqual(mood_source("", ""), "")

    def test_known_words(self):
        label, score = predict_mood(self.model, self.vocab, "круто")
        self.assertEqual(label, "позитив")
        self.assertGreater(score, 0.6)
        label, score = predict_mood(self.model, self.vocab, "хуево")
        self.assertEqual(label, "негатив")
        self.assertGreater(score, 0.6)

    def test_mood_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mood.pt"
            save_mood(path, self.model, self.vocab, notebook_test_acc=1.0)
            loaded, vocab, meta = load_mood(path)
        self.assertEqual(vocab, self.vocab)
        self.assertEqual(meta["notebook_test_acc"], 1.0)
        label, _ = predict_mood(loaded, vocab, "грустно")
        self.assertEqual(label, "негатив")

    def test_replied_message_content(self):
        class Message:
            content = "это ужас"

        class Reference:
            message_id = 7
            resolved = Message()

        class CommandMessage:
            reference = Reference()

        class Ctx:
            message = CommandMessage()

        text = asyncio.run(replied_content(Ctx()))
        self.assertEqual(text, "это ужас")
        label, _ = predict_mood(self.model, self.vocab, text)
        self.assertEqual(label, "негатив")


if __name__ == "__main__":
    unittest.main()
