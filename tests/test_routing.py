import base64
import tempfile
import unittest
from pathlib import Path

from core.bus.queue import JobStore
from core.bus.routing import resolve_worker_kind


class RoutingTests(unittest.TestCase):
    def test_text_models(self):
        self.assertEqual(resolve_worker_kind("text", {"model": "lstm"}), "text_lstm")
        self.assertEqual(
            resolve_worker_kind("text", {"model": "lstm-word"}), "text_lstm_word"
        )
        self.assertEqual(
            resolve_worker_kind("text", {"model": "transformer"}), "text_transformer"
        )
        self.assertEqual(resolve_worker_kind("text", {"model": "gpt"}), "text_gpt")
        self.assertEqual(
            resolve_worker_kind("text", {"model": "deepseek"}), "text_deepseek"
        )
        self.assertEqual(
            resolve_worker_kind("text", {"model": "bert-gen"}), "text_bert_gen"
        )

    def test_mood(self):
        self.assertEqual(resolve_worker_kind("mood", {"source": "hi"}), "mood")

    def test_pic_still_modes(self):
        self.assertEqual(resolve_worker_kind("pic", {"mode_key": "1"}), "pic_still")
        self.assertEqual(resolve_worker_kind("pic", {"mode_key": "2"}), "pic_still")

    def test_pic_mode3_still_vs_video(self):
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
        self.assertEqual(
            resolve_worker_kind("pic", {"mode_key": "3", "animated": False, "data_b64": png}),
            "pic_still",
        )
        self.assertEqual(
            resolve_worker_kind("pic", {"mode_key": "3", "animated": True, "data_b64": png}),
            "pic_video",
        )

    def test_orchestrator_assigns_worker_kind(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "bot.db"
            store = JobStore(db)
            jid = store.enqueue("u1", "text", {"model": "gpt", "start": "Я"})
            self.assertTrue(store.route_one(resolve_worker_kind))
            row = store.get_job(jid)
            self.assertEqual(row["status"], "assigned")
            self.assertEqual(row["worker_kind"], "text_gpt")


if __name__ == "__main__":
    unittest.main()
