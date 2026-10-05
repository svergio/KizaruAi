import inspect
import tempfile
import unittest
from pathlib import Path

from core.bus.queue import JobStore
from core.bus.routing import (
    build_pic_payload,
    build_text_payload,
    pic_mode_name_to_key,
    resolve_worker_kind,
)


class CliRoutingParityTests(unittest.TestCase):
    def test_text_archs_match_bot(self):
        for arch in ("lstm", "transformer", "gpt", "deepseek"):
            payload = build_text_payload(arch, "Привет")
            self.assertEqual(payload, {"model": arch, "start": "Привет"})
            self.assertEqual(resolve_worker_kind("text", payload), f"text_{arch}")
        payload = build_text_payload("bert-gen", "Привет")
        self.assertEqual(resolve_worker_kind("text", payload), "text_bert_gen")

    def test_pic_still_line_cli_export(self):
        payload = build_pic_payload(
            "3",
            b"\x89PNG\r\n\x1a\n",
            "x.png",
            scale=4,
            cells=320,
            animated=False,
        )
        self.assertEqual(resolve_worker_kind("pic", payload), "pic_still")

    def test_pic_still_simple_mode(self):
        payload = build_pic_payload("2", b"data", "x.png")
        self.assertEqual(resolve_worker_kind("pic", payload), "pic_still")

    def test_pic_anim_matches_bot_mode3_gif(self):
        payload = build_pic_payload(
            "3",
            b"GIF89a",
            "clip.gif",
            scale=2,
            cells=160,
            animated=True,
        )
        self.assertEqual(resolve_worker_kind("pic", payload), "pic_video")

    def test_pic_mode_name_keys(self):
        self.assertEqual(pic_mode_name_to_key("line"), "1")
        self.assertEqual(pic_mode_name_to_key("simple"), "2")

    def test_orchestrator_assigns_same_worker_kind_as_cli_payload(self):
        cases = [
            (build_text_payload("transformer"), "text_transformer"),
            (build_pic_payload("1", b"x", "a.png"), "pic_still"),
            (
                build_pic_payload(
                    "3", b"g", "a.gif", scale=2, cells=100, animated=True
                ),
                "pic_video",
            ),
            (
                build_pic_payload(
                    "3",
                    b"\x89PNG\r\n\x1a\n",
                    "a.png",
                    scale=2,
                    cells=100,
                    animated=False,
                ),
                "pic_still",
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            store = JobStore(Path(folder) / "bot.db")
            for index, (payload, expected) in enumerate(cases):
                kind = "text" if "model" in payload else "pic"
                jid = store.enqueue(f"cli:test:{index}", kind, payload)
                self.assertTrue(store.route_one(resolve_worker_kind))
                row = store.get_job(jid)
                self.assertEqual(row["worker_kind"], expected, msg=payload)


class CliEntrypointTests(unittest.TestCase):
    def test_generate_has_no_local_inference(self):
        import app.cli.generate as generate

        src = inspect.getsource(generate)
        self.assertNotIn("evaluate(", src)
        self.assertNotIn("load_model(", src)
        self.assertIn("enqueue_and_wait", src)

    def test_pic_has_no_local_render(self):
        import app.cli.image as pic_cli

        src = inspect.getsource(pic_cli)
        self.assertNotIn("render_pic(", src)
        self.assertNotIn("render_anim_", src)
        self.assertIn("build_pic_payload", src)

    def test_pic_workers_pass_unused_runtime(self):
        import app.workers.still as worker_pic
        import app.workers.video as worker_video

        self.assertIn("job_pic_still(payload, None)", inspect.getsource(worker_pic))
        self.assertIn("job_pic_video(payload, None)", inspect.getsource(worker_video))


if __name__ == "__main__":
    unittest.main()
