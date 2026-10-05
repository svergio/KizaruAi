import unittest
from unittest import mock

from core.bus import handlers
from core.bus.routing import build_text_payload
from core.text.params import (
    CHAR_LENGTH_MAX,
    CHAR_LENGTH_MIN,
    DEFAULTS,
    LINES_MAX,
    TEMP_MAX,
    TEMP_MIN,
    WORD_LENGTH_MAX,
    describe,
    resolve_text_params,
)


class ParamDefaultsTests(unittest.TestCase):
    def test_defaults_from_generation_audit(self):
        self.assertEqual(resolve_text_params("lstm")["temp"], 0.4)
        self.assertEqual(resolve_text_params("transformer")["temp"], 0.9)
        self.assertEqual(resolve_text_params("gpt")["temp"], 1.0)
        self.assertEqual(resolve_text_params("lstm-word")["length"], 72)

    def test_every_model_has_defaults(self):
        for model in DEFAULTS:
            params = resolve_text_params(model)
            self.assertEqual(set(params), {"temp", "length", "lines", "iters"})

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            resolve_text_params("gpt5")

    def test_empty_payload_keeps_defaults(self):
        self.assertEqual(resolve_text_params("gpt", {}), resolve_text_params("gpt"))

    def test_none_values_keep_defaults(self):
        payload = {"temp": None, "lines": None, "length": None, "iters": None}
        self.assertEqual(resolve_text_params("gpt", payload), resolve_text_params("gpt"))


class ParamClampTests(unittest.TestCase):
    def test_temp_clamped_both_ends(self):
        self.assertEqual(resolve_text_params("gpt", {"temp": 99})["temp"], TEMP_MAX)
        self.assertEqual(resolve_text_params("gpt", {"temp": 0})["temp"], TEMP_MIN)

    def test_lines_clamped(self):
        self.assertEqual(resolve_text_params("gpt", {"lines": 500})["lines"], LINES_MAX)
        self.assertEqual(resolve_text_params("gpt", {"lines": -3})["lines"], 1)

    def test_char_and_word_length_have_different_ceilings(self):
        self.assertEqual(
            resolve_text_params("gpt", {"length": 99999})["length"], CHAR_LENGTH_MAX
        )
        self.assertEqual(
            resolve_text_params("lstm-word", {"length": 99999})["length"],
            WORD_LENGTH_MAX,
        )
        self.assertEqual(
            resolve_text_params("gpt", {"length": 1})["length"], CHAR_LENGTH_MIN
        )

    def test_values_in_range_pass_through(self):
        params = resolve_text_params("transformer", {"temp": 0.75, "lines": 4, "length": 200})
        self.assertEqual(params["temp"], 0.75)
        self.assertEqual(params["lines"], 4)
        self.assertEqual(params["length"], 200)


class PayloadTests(unittest.TestCase):
    def test_payload_without_overrides_stays_minimal(self):
        self.assertEqual(
            build_text_payload("gpt", "Привет"), {"model": "gpt", "start": "Привет"}
        )

    def test_payload_carries_only_given_overrides(self):
        payload = build_text_payload("gpt", "Привет", temp=1.2, lines=8)
        self.assertEqual(
            payload, {"model": "gpt", "start": "Привет", "temp": 1.2, "lines": 8}
        )

    def test_describe_mentions_bert_specific_fields(self):
        text = describe("bert-gen", resolve_text_params("bert-gen"))
        self.assertIn("seq_len=", text)
        self.assertIn("iters=", text)
        self.assertNotIn("seq_len=", describe("gpt", resolve_text_params("gpt")))


class HandlerWiringTests(unittest.TestCase):
    def _runtime(self, level="char"):
        model = mock.Mock()
        model.block_size = 256
        return {
            "device": "cpu",
            "level": level,
            "transformer": (model, {}, {}),
            "gpt": (model, {}, {}),
            "deepseek": (model, {}, {}),
            "bert_gen": (model, {}, {}),
            "lstm_model": model,
            "char_to_idx": {},
            "idx_to_char": {},
            "token_to_idx": {},
            "idx_to_token": {},
        }

    def test_lm_handler_forwards_overrides(self):
        payload = build_text_payload("gpt", "Я", temp=0.55, lines=3, length=120)
        with mock.patch.object(handlers, "evaluate_lm", return_value="x") as gen:
            handlers.job_text_gpt(payload, self._runtime())
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["temp"], 0.55)
        self.assertEqual(kwargs["max_lines"], 3)
        self.assertEqual(kwargs["prediction_len"], 120)

    def test_lm_handler_clamps_before_sampling(self):
        payload = build_text_payload("transformer", "Я", temp=50, lines=99)
        with mock.patch.object(handlers, "evaluate_lm", return_value="x") as gen:
            handlers.job_text_transformer(payload, self._runtime())
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["temp"], TEMP_MAX)
        self.assertEqual(kwargs["max_lines"], LINES_MAX)

    def test_lm_handler_defaults_when_not_given(self):
        with mock.patch.object(handlers, "evaluate_lm", return_value="x") as gen:
            handlers.job_text_transformer({"start": "Я"}, self._runtime())
        self.assertEqual(gen.call_args.kwargs["temp"], 0.9)

    def test_char_lstm_handler_forwards_overrides(self):
        payload = build_text_payload("lstm", "Я", temp=0.7, length=400)
        with mock.patch.object(handlers, "evaluate", return_value="x") as gen:
            handlers.job_text_lstm(payload, self._runtime())
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["temp"], 0.7)
        self.assertEqual(kwargs["prediction_len"], 400)

    def test_word_lstm_handler_uses_word_ceiling(self):
        payload = build_text_payload("lstm-word", "Я", length=5000)
        with mock.patch.object(handlers, "evaluate_words", return_value="x") as gen:
            handlers.job_text_lstm_word(payload, self._runtime())
        self.assertEqual(gen.call_args.kwargs["prediction_len"], WORD_LENGTH_MAX)

    def test_bert_gen_handler_forwards_iters_and_caps_window(self):
        payload = build_text_payload("bert-gen", "Я", iters=7, length=1000)
        with mock.patch.object(handlers, "generate_mlm", return_value="x") as gen:
            handlers.job_text_bert_gen(payload, self._runtime())
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["n_iters"], 7)
        self.assertEqual(kwargs["seq_len"], 256)
        self.assertEqual(kwargs["level"], "char")

    def test_bert_gen_handler_passes_word_level(self):
        payload = build_text_payload("bert-gen", "Я", length=48)
        with mock.patch.object(handlers, "generate_mlm", return_value="x") as gen:
            handlers.job_text_bert_gen(payload, self._runtime(level="word"))
        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["level"], "word")
        self.assertEqual(kwargs["seq_len"], 48)

    def test_word_lm_handler_uses_evaluate_lm_words(self):
        payload = build_text_payload("transformer", "Я", length=90)
        with mock.patch.object(handlers, "evaluate_lm_words", return_value="x") as word_gen:
            with mock.patch.object(handlers, "evaluate_lm", return_value="x") as char_gen:
                handlers.job_text_transformer(payload, self._runtime(level="word"))
        self.assertTrue(word_gen.called)
        self.assertFalse(char_gen.called)
        self.assertEqual(word_gen.call_args.kwargs["prediction_len"], 90)

        with mock.patch.object(handlers, "evaluate_lm_words", return_value="x") as word_gen:
            with mock.patch.object(handlers, "evaluate_lm", return_value="x") as char_gen:
                handlers.job_text_gpt({"start": "Я"}, self._runtime(level="char"))
        self.assertTrue(char_gen.called)
        self.assertFalse(word_gen.called)

        with mock.patch.object(handlers, "evaluate_lm_words", return_value="x") as word_gen:
            handlers.job_text_deepseek({"start": "Я"}, self._runtime(level="word"))
        self.assertEqual(word_gen.call_args.kwargs["prediction_len"], 72)


if __name__ == "__main__":
    unittest.main()
