import tempfile
import unittest
from pathlib import Path

import torch

from core.text.bert_gen import BertMLM, apply_mlm_mask, generate_mlm
from core.text.checkpoint import load_model, save_model
from core.text.deepseek import MiniDeepSeek
from core.text.lm import MiniGPT, SimpleTransformer
from core.text.sample import _completed_lyric_lines, evaluate_lm


SAMPLE = (
    "Я живу на районе, cousin, это мой дом.\n"
    "Ночью город не спит, и я снова за рулём.\n"
) * 8


def _vocab(text):
    chars = sorted(set(text))
    char_to_idx = {char: index for index, char in enumerate(chars)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


class LmTests(unittest.TestCase):
    def test_transformer_logits_shape(self):
        char_to_idx, _ = _vocab(SAMPLE)
        model = SimpleTransformer(
            len(char_to_idx), d_model=32, n_layers=2, n_heads=4, block_size=16
        )
        tokens = torch.randint(0, len(char_to_idx), (3, 10))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (3, 10, len(char_to_idx)))

    def test_gpt_logits_shape(self):
        char_to_idx, _ = _vocab(SAMPLE)
        model = MiniGPT(len(char_to_idx), d_model=32, n_layers=2, n_heads=4, block_size=16)
        tokens = torch.randint(0, len(char_to_idx), (2, 8))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, len(char_to_idx)))

    def test_deepseek_logits_shape(self):
        char_to_idx, _ = _vocab(SAMPLE)
        model = MiniDeepSeek(
            len(char_to_idx), d_model=32, n_layers=2, n_heads=4, block_size=16
        )
        tokens = torch.randint(0, len(char_to_idx), (2, 9))
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 9, len(char_to_idx)))

    def test_causal_mask_transformer(self):
        self._assert_causal(SimpleTransformer)

    def test_causal_mask_gpt(self):
        self._assert_causal(MiniGPT)

    def test_causal_mask_deepseek(self):
        self._assert_causal(MiniDeepSeek)

    def _assert_causal(self, cls):
        vocab = 20
        model = cls(vocab, d_model=32, n_layers=2, n_heads=4, block_size=16)
        model.eval()
        left = torch.randint(0, vocab, (1, 8))
        right = left.clone()
        right[0, 5] = (right[0, 5] + 1) % vocab
        with torch.no_grad():
            a = model(left)
            b = model(right)
        self.assertTrue(torch.allclose(a[0, :5], b[0, :5], atol=1e-5))
        self.assertFalse(torch.allclose(a[0, 5:], b[0, 5:], atol=1e-5))

    def test_bert_mlm_is_bidirectional(self):
        vocab = 20
        model = BertMLM(vocab, d_model=32, n_layers=2, n_heads=4, block_size=16)
        model.eval()
        left = torch.randint(0, vocab - 1, (1, 8))
        right = left.clone()
        right[0, 7] = (right[0, 7] + 1) % (vocab - 1)
        with torch.no_grad():
            a = model(left)
            b = model(right)
        self.assertFalse(torch.allclose(a[0, 0], b[0, 0], atol=1e-5))

    def test_mlm_mask_ignores_unmasked(self):
        tokens = torch.randint(0, 10, (2, 8))
        inputs, labels = apply_mlm_mask(tokens, mask_id=10, vocab_size=11, mlm_prob=0.5)
        self.assertEqual(tuple(inputs.shape), tuple(tokens.shape))
        self.assertTrue((labels == -100).any())
        self.assertTrue((labels != -100).any())

    def test_roundtrip_bert_gen(self):
        char_to_idx, idx_to_char = _vocab(SAMPLE)
        model = BertMLM(len(char_to_idx) + 1, d_model=32, n_layers=2, n_heads=4, block_size=16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bert_gen.pt"
            save_model(path, model, char_to_idx, idx_to_char, epoch=1, mean_loss=2.0)
            loaded, _, _, meta = load_model(path, device=torch.device("cpu"))
        self.assertEqual(meta["arch"], "bert_gen")
        self.assertIsInstance(loaded, BertMLM)
        text = generate_mlm(
            loaded, char_to_idx, idx_to_char, start_text="Я", seq_len=12, n_iters=2, temp=0.8
        )
        self.assertTrue(text)

    def test_roundtrip_transformer(self):
        self._assert_roundtrip(SimpleTransformer, "transformer")

    def test_roundtrip_gpt(self):
        self._assert_roundtrip(MiniGPT, "gpt")

    def test_roundtrip_deepseek(self):
        self._assert_roundtrip(MiniDeepSeek, "deepseek")

    def _assert_roundtrip(self, cls, arch):
        char_to_idx, idx_to_char = _vocab(SAMPLE)
        model = cls(len(char_to_idx), d_model=32, n_layers=2, n_heads=4, block_size=16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{arch}.pt"
            save_model(path, model, char_to_idx, idx_to_char, epoch=2, mean_loss=3.1)
            loaded, loaded_chars, loaded_idx, meta = load_model(path, device=torch.device("cpu"))
        self.assertEqual(meta["arch"], arch)
        self.assertEqual(loaded_chars, char_to_idx)
        self.assertIsInstance(loaded, cls)
        text = evaluate_lm(
            loaded, loaded_chars, loaded_idx, start_text="Я", prediction_len=12, temp=0.5
        )
        self.assertTrue(text.startswith("Я"))
        self.assertGreaterEqual(len(text), 8)

    def test_evaluate_lm_max_lines(self):
        char_to_idx, idx_to_char = _vocab(SAMPLE)
        model = SimpleTransformer(
            len(char_to_idx), d_model=32, n_layers=2, n_heads=4, block_size=16
        )
        seed = "я ебал игру как будто шлюху"
        lined = evaluate_lm(
            model,
            char_to_idx,
            idx_to_char,
            start_text=seed,
            prediction_len=400,
            temp=0.8,
            max_lines=6,
        )
        new_lines = _completed_lyric_lines(lined) - _completed_lyric_lines(seed)
        self.assertLessEqual(new_lines, 6)


if __name__ == "__main__":
    unittest.main()
