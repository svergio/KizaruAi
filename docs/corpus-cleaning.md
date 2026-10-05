# Corpus cleaning (Genius → LM)

Training text is built in `training/prepare_data.py` from local `data/corpus/genius_<slug>.json` caches (orchestrator or manual fetch). Those paths are **not shipped in git** — they appear after you download or build corpus. Each song is cleaned before merge into `data/corpus/kizaru_lyrics.txt` (also local-only).

## Pipeline

| Step | Module | What it does |
|------|--------|----------------|
| 1 | `core/corpus/genius.py` | HTML → plain; `<br>` becomes `\n`. Genius often uses double breaks, so raw lyrics are **line + blank line + line**. `strip_page_tail` cuts leaked page JSON when the parser overshoots. |
| 2 | `strip_genius_annotations` (`core/corpus/lyrics.py`) | Drops full-line and leading bracket/paren **section labels** (RU/EN), `[Текст песни «…»]`, contributor/embed UI lines, and similar. |
| 3 | `strip_corpus_noise` (`core/corpus/clean.py`) | **Before `LM_STREAM`:** drops whole songs that still look like Genius page JSON; removes UI/meta lines; strips inline `[?]`, long numeric footnotes, producer credits in brackets, `(Что?)` / `(Read more)` tails, etc. |
| 4 | `_normalize_lyrics_chars` (`core/corpus/lyrics.py`) | Thin spaces, curly quotes, rare symbols. |
| 5 | `normalize_song_body` (`core/corpus/clean.py`) | **LM default:** one file line per verse line. Blank lines inside a song are dropped, the newline between bars stays. Joining bars with a space is opt-in (`intra_line_sep=" "`). |
| 6 | `join_songs` (`core/corpus/clean.py`) | Songs separated by a **single** `\n` (not `\n\n`). |

`_normalize_lyrics_chars` collapses spaces **inside** a line only. A rare symbol used to fall through `" ".join(text.split())`, which also ate newlines and glued a whole song into one line. That path now keeps `\n`. The bot counts a verse by those newlines, so a space-joined song cannot produce six lines.

Optional `NewlineMode.PARAGRAPH` keeps up to two consecutive newlines inside a song (stanza-style); training uses `LM_STREAM`.

Dry run of the current pipeline on 25 cached artists (3 728 songs), nothing written to `kizaru_lyrics.txt`:

| | Raw cache | After `prepare_corpus_text` |
|--|-----------|------------------------------|
| Characters kept | 8.5M | 86% |
| Blank lines | 54% | 1.7% (one trailing newline per song) |
| Lines containing Genius JSON keys | 53 | 0 |
| Mean line length | — | 34 characters |

The live `kizaru_lyrics.txt` is still the older build (blank lines on half the file, page JSON ~11% of characters). Rebuild only after the 300-artist download finishes.

## Noise categories (audit, 25 random `genius_*.json` slugs)

Counts are **songs hit at least once** in the sample (~3k+ tracks). “After” = `prepare_corpus_text(..., genius=True)` with current code.

| Rank | Pattern | Raw (approx.) | After pipeline | Handled by |
|------|---------|---------------|----------------|------------|
| 1 | Bracket section tags (`[Куплет]`, `[Verse]`, `[Припев]`, …) | ~3.3k | ~0 | `strip_genius_annotations` |
| 2 | `[Текст песни «Title»]` title lines | ~3.0k | ~0 | section hint `текст песни` |
| 3 | Short parentheticals in lyrics (ad-libs) | ~2.5k | most kept | intentional (not stripped) |
| 4 | RU sections missing from old hint (`[Рефрен]`, `[Хук]`, `[Интерлюдия:]`, `[Переход:]`, `[Часть I]`) | ~400+ | ~0 | expanded `_is_genius_section_label` |
| 5 | Genius unknown token `[?]` / `[..]` | ~240 | inline removed | `strip_corpus_noise` |
| 6 | `[Label: Artist]` colon headers | ~3.2k raw lines | ~0 when label is section | `_section_label_core` + hints |
| 7 | Page JSON leak (`hotSongsPreview`, `lyricsPlaceholderReason`, …) | ~94 songs in sample | empty song | `corpus_looks_like_json_leak` |
| 8 | Contributors / Translations / English glue lines | rare in RU cache | ~0 | `_GENIUS_NOISE_LINE` + `strip_corpus_noise` |
| 9 | Embed / Read more / You might also like | rare | ~0 | noise line matchers |
| 10 | Producer credits `[Produced by …]` | rare | ~0 | section hint + inline bracket strip |

**Note:** Ad-lib parentheses like `(Эй)`, `(Jillzay, …)`, `(Что?)` **mid-line** in real bars are kept when they are part of a lyric line; standalone `(Что?)` lines and trailing `(Что?)` on a line are removed.

**Existing `kizaru_lyrics.txt`:** built with an older cleaner — lines starting with `[` (e.g. `[Рефрен]`, `[Часть I]`, `[реклама N]`) remain until you rebuild from cache (see below). Inline `[?]` in merged corpus is common.

## When to re-run prepare

- After Genius pool download finishes (orchestrator writes cache only until all slugs are `"complete": true`, then merges corpus).
- After changing `core/corpus/clean.py` or annotation stripping — refresh from cache **without network**:

```bash
python -m training.prepare_data --source genius-cache
```

Do **not** retrain until you intentionally rebuild checkpoints on the new `kizaru_lyrics.txt`.

## After 300-artist orchestrator

1. Wait for `genius_orchestrator` to finish (or confirm all slugs in `GENIUS_ARTISTS` have `genius_<slug>.json` with `"complete": true`).
2. If the orchestrator already wrote `kizaru_lyrics.txt`, run `genius-cache` anyway to apply the latest cleaning rules to the full cache.
3. Optionally run `python -m training.prep_embedding_vocab` if word-level training should match the new corpus (only when you plan training).

See also [runbook.md](runbook.md) (corpus / Genius section).
