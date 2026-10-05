import unittest

from core.corpus.clean import (
    NewlineMode,
    corpus_looks_like_json_leak,
    join_songs,
    normalize_song_body,
    strip_corpus_noise,
)
from core.corpus.lyrics import prepare_corpus_text

GENIUS_LIKE = """
[Текст песни «Test»]


[Куплет 1: Artist]

Первая строка

Вторая строка


[Припев]

Хук один

Хук два
"""

NOISE_SAMPLE = """
144 ContributorsTranslationsEnglish
[Текст песни «Demo»]
[Рефрен: Guest]
Рефрен текст
[Часть II]
Следующая часть
[?]
Не купил я [?] корни, ты знаешь
[Produced by Metro]
Embed
Read More
"""

JSON_LEAK = (
    '","tag":"root"},"lyricsPlaceholderReason":"unreleased",'
    '"hotSongsPreview":[{"url":"https://genius.com/x"}]'
)


class CorpusCleanTests(unittest.TestCase):
    def test_lm_stream_collapses_stanza_gaps(self):
        out = prepare_corpus_text(GENIUS_LIKE, genius=True)
        self.assertNotIn("\n\n", out)
        self.assertIn("Первая строка", out)
        self.assertIn("Вторая строка", out)
        self.assertIn("Хук один", out)
        self.assertNotRegex(out, r"\[Куплет|\[Припев")

    def test_normalize_song_body_keeps_one_line_per_line(self):
        body = "line one\n\nline two\n\n"
        out = normalize_song_body(body, mode=NewlineMode.LM_STREAM)
        self.assertEqual(out, "line one\nline two")

    def test_lm_stream_preserves_line_count(self):
        body = "раз\n\nдва\n\n\nтри\n"
        out = normalize_song_body(body, mode=NewlineMode.LM_STREAM)
        self.assertEqual(out.split("\n"), ["раз", "два", "три"])
        self.assertNotIn("\n\n", out)

    def test_rare_symbol_does_not_glue_lines(self):
        text = "первая строка ★\nвторая строка\nтретья строка"
        out = prepare_corpus_text(text)
        self.assertEqual(
            [line.strip() for line in out.splitlines() if line.strip()],
            ["первая строка", "вторая строка", "третья строка"],
        )
        self.assertNotIn("★", out)

    def test_space_join_still_available_explicitly(self):
        body = "line one\n\nline two"
        out = normalize_song_body(body, mode=NewlineMode.LM_STREAM, intra_line_sep=" ")
        self.assertEqual(out, "line one line two")

    def test_join_songs_single_newline(self):
        out = join_songs(["song a", "song b"], separator="\n")
        self.assertEqual(out, "song a\nsong b\n")

    def test_prepare_corpus_strips_noise_and_placeholders(self):
        out = prepare_corpus_text(NOISE_SAMPLE, genius=True)
        self.assertNotIn("Contributors", out)
        self.assertNotIn("Embed", out)
        self.assertNotIn("Produced", out)
        self.assertNotIn("[Рефрен", out)
        self.assertNotIn("[Часть", out)
        self.assertNotIn("[?]", out)
        self.assertIn("Рефрен текст", out)
        self.assertIn("корни", out)

    def test_strip_corpus_noise_jsonish_line(self):
        out = strip_corpus_noise('{"view_song": true}\nReal line')
        self.assertIn("Real line", out)
        self.assertNotIn("view_song", out)

    def test_prepare_drops_json_leak_song(self):
        out = prepare_corpus_text(JSON_LEAK, genius=True)
        self.assertEqual(out, "")

    def test_json_leak_detector(self):
        self.assertTrue(corpus_looks_like_json_leak(JSON_LEAK))
        self.assertFalse(corpus_looks_like_json_leak("normal line\nanother line"))

    def test_refrain_and_interlude_labels_removed(self):
        raw = """
[Интерлюдия: MC]
Речь в интерлюдии
[Переход: DJ]
Бит крутится
[Хук]
Мы на районе
"""
        out = prepare_corpus_text(raw, genius=True)
        self.assertNotRegex(out, r"\[Интерлюд|\[Переход|\[Хук")
        self.assertIn("Речь в интерлюдии", out)
        self.assertIn("Мы на районе", out)


if __name__ == "__main__":
    unittest.main()
