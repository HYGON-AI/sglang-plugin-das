# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch")
class Kimi31DiagnosticsTests(unittest.TestCase):
    def test_finite_module_is_transparent_and_install_is_idempotent(self):
        from hcu_sglang.kimi31_diagnostics import _install_model

        class KimiK3MLP(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states + 1

        model = KimiK3MLP()
        with contextlib.redirect_stdout(io.StringIO()):
            _install_model(model)
            _install_model(model)
        value = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.equal(model(hidden_states=value), value + 1))
        self.assertEqual(len(model._forward_hooks), 1)
        self.assertEqual(len(model._forward_pre_hooks), 1)

    def test_nonfinite_input_stops_before_module_and_preserves_input(self):
        from hcu_sglang.kimi31_diagnostics import _install_model

        calls = []

        class KimiK3MLP(torch.nn.Module):
            def forward(self, hidden_states):
                calls.append(True)
                return hidden_states

        model = KimiK3MLP()
        value = torch.tensor([[1.0, float("nan")]])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _install_model(model)
            with self.assertRaisesRegex(RuntimeError, r"\.input"):
                model(hidden_states=value)
        record = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(record["nan"], 1)
        self.assertEqual(record["bad_rows"], [0])
        self.assertEqual(calls, [])
        self.assertTrue(torch.isnan(value[0, 1]))

    def test_nonfinite_output_reports_producer(self):
        from hcu_sglang.kimi31_diagnostics import _install_model

        class KimiK3MLP(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states * float("nan")

        model = KimiK3MLP()
        with contextlib.redirect_stdout(io.StringIO()):
            _install_model(model)
            with self.assertRaisesRegex(RuntimeError, r"\.output"):
                model(torch.ones(2, 3))

    def test_logits_allow_masked_negative_infinity_but_reject_nan_and_posinf(self):
        from hcu_sglang.kimi31_diagnostics import _check

        _check(torch.tensor([[0.0, -float("inf")]]), "logits", logits=True)
        for invalid in (float("nan"), float("inf")):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    _check(torch.tensor([[0.0, invalid]]), "logits", logits=True)

    def test_dump_retains_state_and_marks_input_timing(self):
        from hcu_sglang.kimi31_diagnostics import _dump

        module = torch.nn.Linear(2, 2)
        value = torch.ones(1, 2)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"HCU_KIMI31_DIAG_DIR": directory}):
                with contextlib.redirect_stdout(io.StringIO()):
                    _dump(module, (value,), {}, value, "layer.output")
            files = list(Path(directory).glob("*.pt"))
            self.assertEqual(len(files), 1)
            saved = torch.load(files[0], weights_only=True)
        self.assertTrue(torch.equal(saved["args"][0], value))
        self.assertTrue(torch.equal(saved["state_dict"]["weight"], module.weight))
        self.assertIn("post-forward", saved["input_snapshot_timing"])

    def test_nonfinite_probabilities_never_reach_multinomial(self):
        from hcu_sglang.kimi31_diagnostics import _multinomial

        calls = []
        for value in ([[float("nan")]], [[-1.0, 2.0]], [[0.0, 0.0]]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    _multinomial(lambda *a, **k: calls.append(True), torch.tensor(value))
        self.assertEqual(calls, [])
        valid = torch.tensor([[0.25, 0.75]])
        self.assertIs(_multinomial(lambda p: p, valid), valid)


if __name__ == "__main__":
    unittest.main()
