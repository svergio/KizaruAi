import re
from enum import Enum


class NewlineMode(str, Enum):
    # LM_STREAM: одна строка песни = одна строка файла, пустых строк нет.
    # Перенос строки — единственный сигнал конца строки куплета, по нему бот и считает 6 строк.
    LM_STREAM = "lm_stream"
    PARAGRAPH = "paragraph"


_CORPUS_JSON_LEAK = re.compile(
    r"(?i)"
    r'"(?:tag|lyricsPlaceholderReason|hotSongsPreview|defaultQuestions|eligibleForIqByLyricsSize)"'
    r'|add_pinned_annotation|"view_(?:apple_music|song|recommendations)"'
    r'|"children"\s*:|window\.__'
)

_JSONISH_LINE = re.compile(
    r"(?i)"
    r'^\s*[\[{"]'
    r'|"\w+"\s*:'
    r"|hotSongsPreview|lyricsPlaceholderReason|defaultQuestions"
    r'|"view_\w+"|add_pinned_annotation'
)

_GENIUS_UI_LINE = re.compile(
    r"(?i)^\s*(?:"
    r"\d+\s+contributors?(?:\s*and\s+translations?|\s*\/\s*)?(?:\s*\w+)*\s*$|"
    r"(?:\d+\s+)?contributors?(?:translations?|english|russian)|"
    r"you might also like\s*$|"
    r"read\s+more\s*$|"
    r"embed\s*$|"
    r"^\s*translations?\s*$|"
    r"^\s*english\s*$|"
    r"^\s*russian\s*$|"
    r"^\s*\d+\s+embed\s*$"
    r")"
)

_LYRICS_TITLE_LINE = re.compile(r"(?i)^\s*.+\s+Lyrics\s*$")

_BRACKET_INLINE_NOISE = re.compile(
    r"(?i)\[(?:"
    r"(?:Produced|Written|Mixed|Mastered|Recorded)\s+by[^\]]*"
    r"|\?"
    r"|\.\.+|…+"
    r"|\d{6,}"
    r")\]"
)

_BRACKET_VERSION_TAG = re.compile(
    r"(?i)\[(?:acoustic version|live version|remix|bonus track|deluxe edition)\]"
)

_PAREN_LINE_NOISE = re.compile(
    r"(?i)^\s*\((?:"
    r"Produced|Prod\.|Written|Mixed|Mastered|Recorded)\s+by[^)]+\)\s*$"
)

_TRAILING_GENIUS_PAREN = re.compile(r"(?i)\s*\((?:Что\?|What\?|Read\s+more)\)\s*")


def corpus_looks_like_json_leak(text):
    if not text or not text.strip():
        return False
    if len(text) < 60:
        return False
    return bool(_CORPUS_JSON_LEAK.search(text))


def _strip_inline_bracket_noise(line):
    line = _BRACKET_VERSION_TAG.sub("", line)
    line = re.sub(r"\[\?\]", "", line)
    line = re.sub(r"\[\.\.+\]", "", line)
    line = re.sub(r"\[…+\]", "", line)
    prev = None
    while prev != line:
        prev = line
        line = _BRACKET_INLINE_NOISE.sub("", line)
        line = re.sub(r"\[\d{6,}\]", "", line)
    line = _TRAILING_GENIUS_PAREN.sub("", line)
    return " ".join(line.split())


def strip_corpus_noise(text):
    if not text or not text.strip():
        return ""
    if corpus_looks_like_json_leak(text):
        return ""
    kept = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if _JSONISH_LINE.search(line):
            continue
        if _GENIUS_UI_LINE.match(line):
            continue
        low = line.lower()
        if low in ("embed", "read more") or "you might also like" in low:
            continue
        if _LYRICS_TITLE_LINE.match(line) and not re.search(r"[\u0400-\u04FF]", line):
            continue
        if _PAREN_LINE_NOISE.match(line):
            continue
        if re.match(r"(?i)^\s*\((?:Что\?|What\?)\)\s*$", line):
            continue
        line = _strip_inline_bracket_noise(line)
        if not line:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def collapse_consecutive_newlines(text, max_consecutive=1):
    if max_consecutive < 1:
        max_consecutive = 1
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    replacement = "\n" * max_consecutive
    return re.sub(pattern, replacement, text)


def strip_empty_lines(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def trim_lines(text):
    return "\n".join(line.strip() for line in text.splitlines())


def normalize_song_body(text, mode=NewlineMode.LM_STREAM, intra_line_sep="\n"):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = trim_lines(text)
    non_empty = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty:
        return ""
    if mode == NewlineMode.PARAGRAPH:
        body = "\n".join(non_empty)
        return collapse_consecutive_newlines(body, max_consecutive=2)
    sep = intra_line_sep if intra_line_sep is not None else " "
    return sep.join(non_empty)


def join_songs(parts, separator="\n"):
    kept = [p.strip() for p in parts if (p or "").strip()]
    if not kept:
        return ""
    out = separator.join(kept)
    if not out.endswith("\n"):
        out += "\n"
    return out


def normalize_corpus_text(text, mode=NewlineMode.LM_STREAM, intra_line_sep="\n"):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n{2,}", text)
    normalized = []
    for chunk in chunks:
        piece = normalize_song_body(chunk, mode=mode, intra_line_sep=intra_line_sep)
        if piece:
            normalized.append(piece)
    return join_songs(normalized, separator="\n")
