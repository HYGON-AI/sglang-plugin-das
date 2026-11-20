"""Fused SiLU*up + INT8 per-token quant for the W4A8 MoE activation.

The flag lives in a module whose import pulls in the whole SGLang quantization
stack, so this loads the gate expression from the real source rather than
importing the module. The op's bitwise equivalence and speed are device
measurements recorded in docs/hcu-sglang-development.md, not testable here.
"""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hcu_sglang/layers/quantization/slimquant_w4a8_marlin.py"
FLAG = "W4A8_TPMOE_FUSED_ACT_QUANT"
ENV = "SGLANG_W4A8_TPMOE_FUSED_ACT_QUANT"


def load_gate_expression():
    """Return the source text of the `FLAG = ...` assignment."""
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == FLAG
        ):
            return ast.get_source_segment(SOURCE.read_text(), node)
    raise AssertionError(f"{FLAG} assignment not found in {SOURCE}")


def evaluate_gate(env_value):
    expression = load_gate_expression()
    env = dict(os.environ)
    env.pop(ENV, None)
    if env_value is not None:
        env[ENV] = env_value
    scope = {"os": os}
    with patch.dict(os.environ, env, clear=True):
        exec(expression, scope)
    return scope[FLAG]


class FusedActQuantGateTests(unittest.TestCase):
    def test_default_is_off(self):
        self.assertEqual(evaluate_gate(None), False)

    def test_only_the_value_one_enables_it(self):
        for value, expected in (("1", True), ("0", False), ("true", False),
                                ("yes", False), ("", False), ("2", False)):
            self.assertEqual(evaluate_gate(value), expected, repr(value))



class FusedActQuantWiringTests(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text()

    def test_the_op_is_optional_and_falls_back(self):
        # Imported under a try/except so a lightop without the op keeps working.
        self.assertIn("from lightop import fuse_silu_mul_per_token_quant", self.text)
        self.assertIn("_fuse_act_quant = None", self.text)
        self.assertIn("fuse_silu_mul_per_token_quant = _fuse_act_quant", self.text)

    def test_gelu_stays_on_the_two_step_path(self):
        # The fused op only exists for SiLU; the gate must name the activation.
        marker = "activation == \"silu\""
        self.assertIn(marker, self.text)
        self.assertIn("elif activation == \"gelu\":", self.text)

    def test_fallback_reproduces_the_original_two_steps(self):
        self.assertIn("F.silu(gate) * up", self.text)
        self.assertIn("per_token_quant_int8(", self.text)


if __name__ == "__main__":
    unittest.main()
