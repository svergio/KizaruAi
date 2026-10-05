import inspect
import tempfile
import unittest
from pathlib import Path


class BotIOTests(unittest.TestCase):
    def test_bot_has_no_torch_generation(self):
        import app.discord.__main__ as discord_app

        src = inspect.getsource(discord_app)
        self.assertNotIn("evaluate(", src)
        self.assertNotIn("evaluate_lm(", src)
        self.assertNotIn("render_pic(", src)
        self.assertNotIn("load_model(", src)

    def test_build_bot_slash_surface(self):
        from app.discord.__main__ import build_bot

        with tempfile.TemporaryDirectory() as folder:
            bot = build_bot(db_path=Path(folder) / "bot.db")
            group = bot.tree.get_command("ai")
            names = {command.name for command in group.commands}
            self.assertEqual(names, {"text", "mood", "pic", "help"})


if __name__ == "__main__":
    unittest.main()
