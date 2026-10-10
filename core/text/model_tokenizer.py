"""Model vocab as HF-like tokenizer (encode -> list[int]) for token counts."""


class ModelVocabTokenizer:
    def __init__(self, vocab, *, level: str = "char", idx_to_token=None, bpe_tokenizer=None):
        self.vocab = vocab
        self.level = level
        self.idx_to_token = idx_to_token or {}
        self.bpe_tokenizer = bpe_tokenizer

    def encode(self, text: str) -> list[int]:
        text = str(text)
        if self.level == "bpe":
            if self.bpe_tokenizer is None:
                raise RuntimeError("bpe tokenizer missing")
            from core.text.bpe import encode_ids

            return encode_ids(self.bpe_tokenizer, text)
        if self.level == "word":
            from core.text.tokenize import encode_words

            return list(encode_words(text, self.vocab))
        return [self.vocab[ch] for ch in text if ch in self.vocab]
