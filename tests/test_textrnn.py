import unittest
from pathlib import Path

import torch

from kizaru_rnn.data import get_batch, load_corpus, text_to_seq
from kizaru_rnn.generate import evaluate
from kizaru_rnn.model import TextRNN


SAMPLE = (
    "Я живу на районе, cousin, это мой дом.\n"
    "Ночью город не спит, и я снова за рулём.\n"
    "Hey, keep it real, не забудь про свой путь.\n"
) * 40


class TextRNNTests(unittest.TestCase):
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
        train, target = get_batch(sequence, seq_len=32, batch_size=4)
        train = train.permute(1, 0, 2)
        hidden = model.init_hidden(4)
        output, hidden_out = model(train, hidden)
        self.assertEqual(tuple(output.shape), (31, 4, len(idx_to_char)))
        self.assertEqual(tuple(hidden_out[0].shape), (2, 4, 32))

        text = evaluate(
            model,
            char_to_idx,
            idx_to_char,
            start_text="Я",
            prediction_len=20,
            temp=0.8,
        )
        self.assertTrue(text.startswith("Я"))
        self.assertEqual(len(text), 21)

    def test_legacy_messages_loadable(self):
        path = Path("legacy/messages.csv")
        self.assertTrue(path.exists())
        text = load_corpus(path)
        self.assertGreater(len(text), 1000)


if __name__ == "__main__":
    unittest.main()
