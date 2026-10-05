import json
import tempfile
import unittest
from pathlib import Path

from core.corpus.genius import cache_path
from core.corpus.genius_pool import (
    MAX_WORKERS,
    adopt_legacy_caches,
    clamp_workers,
    pending_slugs,
)


class GeniusPoolTests(unittest.TestCase):
    def test_clamp_workers_never_above_five(self):
        self.assertEqual(clamp_workers(0), 1)
        self.assertEqual(clamp_workers(5), 5)
        self.assertEqual(clamp_workers(12), MAX_WORKERS)

    def test_pending_skips_only_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            done = cache_path(cache, "Kizaru")
            done.write_text(
                json.dumps({"slug": "Kizaru", "songs": [{"text": "a"}], "complete": True}),
                encoding="utf-8",
            )
            partial = cache_path(cache, "Mayot")
            partial.write_text(
                json.dumps({"slug": "Mayot", "songs": [{"text": "b"}], "complete": False}),
                encoding="utf-8",
            )
            waiting = pending_slugs(("Kizaru", "Mayot", "Guf"), cache)
            self.assertEqual(waiting, ["Mayot", "Guf"])

    def test_adopt_stamps_finished_and_drops_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            cache_path(cache, "ATL").write_text(
                json.dumps({"slug": "ATL", "songs": [{"text": "line"}]}),
                encoding="utf-8",
            )
            cache_path(cache, "Mayot").write_text(
                json.dumps({"slug": "Mayot", "songs": [{"text": "cut"}]}),
                encoding="utf-8",
            )
            stamped, dropped = adopt_legacy_caches(cache, partial_slugs=("Mayot",))
            self.assertEqual(stamped, ["ATL"])
            self.assertEqual(dropped, ["Mayot"])
            self.assertFalse(cache_path(cache, "Mayot").exists())
            kept = json.loads(cache_path(cache, "ATL").read_text(encoding="utf-8"))
            self.assertTrue(kept["complete"])
            self.assertEqual(pending_slugs(("ATL", "Mayot"), cache), ["Mayot"])


class ArtistPickTests(unittest.TestCase):
    def test_exact_name_not_substring(self):
        from core.corpus.genius import pick_artist_hit

        hits = [
            {"id": 1, "name": "Guffern", "slug": "Guffern"},
            {"id": 2, "name": "Guf", "slug": "Guf"},
        ]
        chosen = pick_artist_hit(hits, "Guf")
        self.assertEqual(chosen["id"], 2)
        self.assertIsNone(pick_artist_hit(hits[:1], "Guf"))

    def test_hyphen_matches_spaced_name(self):
        from core.corpus.genius import pick_artist_hit

        hits = [{"id": 9, "name": "Andy Panda", "slug": "Andy-panda"}]
        chosen = pick_artist_hit(hits, "Andy-Panda")
        self.assertEqual(chosen["id"], 9)


class ArtistPageParseTests(unittest.TestCase):
    def test_reads_id_next_to_exact_slug(self):
        from core.corpus.genius import parse_artist_page

        page = (
            '\\"slug\\":\\"Guffern\\",\\"id\\":1 '
            '\\"slug\\":\\"Guf\\",\\"name\\":\\"Guf\\",\\"id\\":34707'
        )
        self.assertEqual(parse_artist_page(page, "Guf"), (34707, "Guf"))
        lsp = '\\"slug\\":\\"Lsp\\",\\"name\\":\\"LSP\\",\\"id\\":143013'
        self.assertEqual(parse_artist_page(lsp, "LSP"), (143013, "LSP"))


class StripPageTailTests(unittest.TestCase):
    def test_cuts_children_and_keeps_lines(self):
        from core.corpus.genius import lyrics_looks_clean, strip_page_tail

        raw = 'куплет один\\nкуплет два\n\n","children":[{"children":["x"]}'
        text = strip_page_tail(raw)
        self.assertEqual(text, "куплет один\nкуплет два")
        self.assertTrue(lyrics_looks_clean(text))


class DirtyLyricsTests(unittest.TestCase):
    def test_page_dump_is_not_clean_and_not_complete(self):
        from core.corpus.genius import lyrics_looks_clean
        from core.corpus.genius_pool import cache_is_complete

        self.assertTrue(lyrics_looks_clean("я еду по району и держу свой flow весь куплет тут"))
        self.assertFalse(lyrics_looks_clean('кусок {"children":[]} window.__APP'))
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache_path(cache, "Guf")
            path.write_text(
                json.dumps(
                    {
                        "slug": "Guf",
                        "complete": True,
                        "songs": [
                            {
                                "title": "x",
                                "text": 'норма не норма а {"children":[1]} и ещё текст страницы',
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(cache_is_complete(path))
            self.assertEqual(pending_slugs(("Guf",), cache), ["Guf"])


if __name__ == "__main__":
    unittest.main()
