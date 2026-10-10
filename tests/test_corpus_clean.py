"""core/corpus/clean.py: чистка шума Genius, тегов [..] и нормализация переводов строк."""
from core.corpus.clean import (
    NewlineMode,
    collapse_consecutive_newlines,
    corpus_looks_like_json_leak,
    join_songs,
    normalize_corpus_text,
    normalize_song_body,
    strip_corpus_noise,
    strip_empty_lines,
    trim_lines,
)


def test_json_leak_detected_and_dropped():
    text = '{"tag": "root", "children": [{"x": 1}]} ' + "мусор " * 20
    assert corpus_looks_like_json_leak(text)
    assert strip_corpus_noise(text) == ""


def test_short_or_empty_text_is_not_json_leak():
    assert not corpus_looks_like_json_leak("")
    assert not corpus_looks_like_json_leak('"tag"')


def test_genius_ui_lines_removed():
    raw = "12 Contributors\nЯ иду по району\nYou might also like\nEmbed\nRead More\nНочь тиха"
    assert strip_corpus_noise(raw) == "Я иду по району\nНочь тиха"


def test_produced_by_lines_removed():
    raw = "Produced by Slava Marlow\n(Produced by Kizaru)\nЯ на месте"
    assert strip_corpus_noise(raw) == "Я на месте"


def test_inline_bracket_noise_removed():
    raw = "Я тут [?] стою [1234567]\nПесня [Remix]\nДальше [...] едем"
    assert strip_corpus_noise(raw) == "Я тут стою\nПесня\nДальше едем"


def test_english_title_lyrics_line_removed():
    raw = "Money Talks Lyrics\nДеньги говорят"
    assert strip_corpus_noise(raw) == "Деньги говорят"


def test_jsonish_line_removed():
    raw = 'Строка песни\n"foo": "bar"\nЕщё строка'
    assert strip_corpus_noise(raw) == "Строка песни\nЕщё строка"


def test_strip_noise_keeps_plain_lyrics():
    raw = "Кайф и огонь\nНочь, город, дым!"
    assert strip_corpus_noise(raw) == raw


def test_strip_noise_empty_input():
    assert strip_corpus_noise("") == ""
    assert strip_corpus_noise("   \n  ") == ""


def test_collapse_consecutive_newlines():
    assert collapse_consecutive_newlines("a\n\n\n\nb", 2) == "a\n\nb"
    assert collapse_consecutive_newlines("a\r\n\r\nb", 1) == "a\nb"
    assert collapse_consecutive_newlines("a\n\n\nb", 0) == "a\nb"


def test_strip_empty_and_trim_lines():
    assert strip_empty_lines("a\n\n  \nb") == "a\nb"
    assert trim_lines("  a  \n b ") == "a\nb"


def test_normalize_song_body_modes():
    text = "  первая \r\n\r\n вторая  \n\n\nтретья "
    assert normalize_song_body(text) == "первая\nвторая\nтретья"
    assert normalize_song_body(text, intra_line_sep=" ") == "первая вторая третья"
    assert normalize_song_body(text, mode=NewlineMode.PARAGRAPH) == "первая\nвторая\nтретья"
    assert normalize_song_body("   \n ") == ""


def test_normalize_corpus_text_splits_songs_on_blank_lines():
    text = "песня один\nстрока два\n\n\nпесня два\n"
    assert normalize_corpus_text(text) == "песня один\nстрока два\nпесня два\n"


def test_join_songs():
    assert join_songs([" a ", "", None, "b"]) == "a\nb\n"
    assert join_songs([]) == ""
    assert join_songs(["x"], separator="\n\n") == "x\n"
