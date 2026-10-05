import json
import unittest
from pathlib import Path

from core.corpus.pack import row_has_section_tag_line

DATASET = Path("data/dataset/lyrics.jsonl")


class DatasetRowTests(unittest.TestCase):
    def test_no_bracket_section_tags_in_dataset(self):
        if not DATASET.exists():
            self.skipTest("run pack_lyrics_dataset.py first")
        bad = []
        with DATASET.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = json.loads(line)
                text = row.get("text") or ""
                if row_has_section_tag_line(text):
                    bad.append((line_no, row.get("id")))
        self.assertEqual(bad, [], msg=f"section tags in rows: {bad[:5]}")


if __name__ == "__main__":
    unittest.main()
