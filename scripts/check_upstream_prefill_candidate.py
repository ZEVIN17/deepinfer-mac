"""Standalone dynamic-prefill regression; no DeepInfer imports or downloaded weights.

Run without --source to demonstrate missing API on installed MLX-LM 0.32.2.
Pass a patched generate.py to test only its BatchGenerator against installed
MLX-LM primitives. Full upstream CI is a separate requirement before submission.
"""
import argparse
import ast
import importlib
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path)
args = parser.parse_args()
generate = importlib.import_module('mlx_lm.generate')
Generator = generate.BatchGenerator
if args.source:
    tree = ast.parse(args.source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchGenerator')
    namespace = dict(vars(generate))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.source), 'exec'), namespace)  # noqa: S102 — explicit local source regression runner
    Generator = namespace['BatchGenerator']


class PrefillSetterTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.model = Model(ModelArgs(model_type='llama', hidden_size=32, num_hidden_layers=2,
                                    intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                                    rms_norm_eps=1e-5, vocab_size=32))
        self.gen = Generator(self.model, prefill_step_size=2, prefill_batch_size=1, completion_batch_size=2)

    def tearDown(self):
        mx.synchronize(self.gen.stream)
        self.gen.close()

    def test_inflight_and_future_batch(self):
        uid = self.gen.insert([[1, 2, 3, 4, 5, 6, 7]], max_tokens=[3])[0]
        self.assertEqual(self.gen.next()[0][0].progress[0], 2)
        self.gen.set_prefill_step_size(1)
        self.assertEqual(self.gen.next()[0][0].progress[0], 3)
        self.gen.remove([uid])
        self.gen.set_prefill_step_size(3)
        self.gen.insert([[1, 2, 3, 4, 5, 6, 7]], max_tokens=[3])
        self.assertEqual(self.gen.next()[0][0].progress[0], 3)
        self.assertEqual(self.gen._prompt_batch.prefill_step_size, 3)

    def test_invalid_value_is_atomic(self):
        for value in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError):
                self.gen.set_prefill_step_size(value)
            self.assertEqual(self.gen.prefill_step_size, 2)
            self.assertEqual(self.gen._prompt_batch.prefill_step_size, 2)

    def test_valid_update_on_empty_batch(self):
        self.gen.set_prefill_step_size(8)
        self.assertEqual(self.gen.prefill_step_size, 8)
        self.assertEqual(self.gen._prompt_batch.prefill_step_size, 8)


if __name__ == '__main__':
    unittest.main(argv=['prefill-setter-regression'])
