import re
import unittest

from core.corpus.lyrics import prepare_corpus_text, strip_genius_annotations

SAMPLE = """
144 ContributorsTranslationsEnglish
[Текст песни «MONEY LONG»]

[Куплет 1: Kizaru]
Я еду по району, cousin, это мой дом
[Припев] я ебал этот beat, держу свой flow

[Verse 2]
Hey, keep it real

[Chorus]
We don't stop

Embed
You might also like
Read More
"""


class GeniusCleanTests(unittest.TestCase):
    def test_strip_annotations_keeps_lyrics(self):
        out = strip_genius_annotations(SAMPLE)
        self.assertNotRegex(out, r"\[(?:Verse|Припев|Куплет|Chorus)")
        self.assertIn("я еду по району", out.lower())
        self.assertIn("я ебал этот beat", out.lower())
        self.assertIn("keep it real", out.lower())
        self.assertNotIn("Contributors", out)
        self.assertNotIn("Embed", out)
        self.assertNotIn("You might also like", out)

    def test_prepare_corpus_text_end_to_end(self):
        out = prepare_corpus_text(SAMPLE, genius=True)
        self.assertNotRegex(out, r"\[Verse|\[Припев|\[Куплет|\[Chorus")
        self.assertIn("я ебал", out)


if __name__ == "__main__":
    unittest.main()
