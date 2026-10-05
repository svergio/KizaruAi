import unittest

from core.text.tokenize import (
    TOKEN_NL,
    TOKEN_UNK,
    build_word_vocab,
    detokenize,
    encode_words,
    text_to_word_seq,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    def test_tokenize_keeps_words_and_newlines(self):
        tokens = tokenize("Я еду\nпо району")
        self.assertEqual(tokens, ["я", "еду", TOKEN_NL, "по", "району"])

    def test_detokenize_roundtrip_shape(self):
        text = detokenize(["я", "еду", ",", TOKEN_NL, "cousin"])
        self.assertIn("я еду,", text)
        self.assertIn("\ncousin", text)

    def test_vocab_min_freq_and_unk(self):
        corpus = ("слово " * 6) + "редкое хаб"
        token_to_idx, _ = build_word_vocab(corpus, min_freq=3)
        self.assertIn("слово", token_to_idx)
        self.assertNotIn("редкое", token_to_idx)
        ids = encode_words("слово редкое", token_to_idx)
        self.assertEqual(ids[0], token_to_idx["слово"])
        self.assertEqual(ids[1], token_to_idx[TOKEN_UNK])

    def test_text_to_word_seq_length(self):
        sequence, token_to_idx, idx_to_token = text_to_word_seq(
            "раз два раз два раз\nтри", min_freq=1
        )
        self.assertGreaterEqual(len(sequence), 5)
        self.assertEqual(idx_to_token[token_to_idx[TOKEN_NL]], TOKEN_NL)

    def test_whitelist_keeps_rare_and_seed_only(self):
        corpus = ("слово " * 6) + "редкое"
        token_to_idx, _ = build_word_vocab(
            corpus, min_freq=3, whitelist=["редкое", "братва"]
        )
        self.assertIn("редкое", token_to_idx)
        self.assertIn("братва", token_to_idx)
        ids = encode_words("братва редкое", token_to_idx)
        self.assertNotEqual(ids[0], token_to_idx[TOKEN_UNK])
        self.assertNotEqual(ids[1], token_to_idx[TOKEN_UNK])


if __name__ == "__main__":
    unittest.main()
