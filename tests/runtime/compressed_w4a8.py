from copy import deepcopy
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock, patch

import torch
from compressed_tensors.quantization import QuantizationArgs
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
from hcu_sglang.environ import register_envs

bootstrap_aiter_compat()
register_envs()

from hcu_sglang.layers.quantization import compressed_w4a8 as C
from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as W
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe import utils as U
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from sglang.srt.plugins.hook_registry import HookRegistry


def scheme():
    return {
        "weights": QuantizationArgs(num_bits=4, type="int", strategy="channel", symmetric=True, dynamic=False),
        "input_activations": QuantizationArgs(num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True),
    }


class CompressedW4A8Tests(unittest.TestCase):
    def test_initializer_preserves_actual_activation_and_other_groups(self):
        entry = scheme()
        activation = entry["input_activations"]
        plain = {"weights": QuantizationArgs(num_bits=8, type="int", strategy="channel"), "input_activations": None}
        rows = {"re:.*w[123]$": entry, "dense": plain}
        config = S()
        def original(obj, target_scheme_map, ignore):
            obj.target_scheme_map = target_scheme_map
            target_scheme_map["re:.*w[123]$"]["input_activations"] = None
            return 17
        with patch.object(C, "is_hcu", return_value=True):
            self.assertEqual(C.around_init(original, config, rows, ["re:.*engram.*"]), 17)
        self.assertIs(config.target_scheme_map["re:.*w[123]$"]["input_activations"], activation)
        self.assertIs(config.target_scheme_map["dense"], plain)

    def test_initializer_keeps_non_hcu_baseline(self):
        rows = {"re:.*w[123]$": scheme()}
        def original(obj, target_scheme_map, ignore):
            obj.target_scheme_map = target_scheme_map
            target_scheme_map["re:.*w[123]$"]["input_activations"] = None
        with patch.object(C, "is_hcu", return_value=False):
            C.around_init(original, S(), rows, ["engram"])
        self.assertIsNone(rows["re:.*w[123]$"]["input_activations"])

    def test_projection_alias_and_consistency(self):
        q = scheme()
        config = S(target_scheme_map={"w[123]": q}, ignore=["engram"], _add_fused_moe_to_target_scheme_map=Mock())
        names = []
        def lookup(layer, name):
            names.append(name)
            if "gate_proj" in name:
                raise ValueError("alias")
            return q
        config.get_scheme_dict = lookup
        self.assertIs(C._projection_scheme(config, object(), "layers.0.mlp.experts"), q)
        self.assertEqual(names[-3:], ["layers.0.mlp.experts.0.w1", "layers.0.mlp.experts.0.w3", "layers.0.mlp.experts.0.w2"])
        config.get_scheme_dict = Mock(side_effect=[q, None, q])
        with self.assertRaisesRegex(ValueError, "same quantization"):
            C._projection_scheme(config, object(), "x")

    def test_other_projection_errors_are_preserved(self):
        error = ValueError("unmatched")
        config = S(target_scheme_map={}, ignore=[], _add_fused_moe_to_target_scheme_map=Mock(), get_scheme_dict=Mock(side_effect=error))
        with self.assertRaises(ValueError) as got:
            C._projection_scheme(config, object(), "x")
        self.assertIs(got.exception, error)

    def test_unsupported_quantization_is_not_selected(self):
        self.assertTrue(C._is_channel_w4a8(scheme()))
        for key, field, value in [("weights", "num_bits", 8), ("weights", "symmetric", False), ("weights", "dynamic", True), ("weights", "strategy", "group"), ("input_activations", "num_bits", 4), ("input_activations", "strategy", "tensor"), ("input_activations", "dynamic", False), ("input_activations", "symmetric", False)]:
            with self.subTest(key=key, field=field, value=value):
                row = scheme()
                setattr(row[key], field, value)
                self.assertFalse(C._is_channel_w4a8(row))
        self.assertFalse(C._is_channel_w4a8(None))

    def test_actual_config_public_consumer_and_hook_registration(self):
        import json
        from pathlib import Path
        raw = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "deepseek_v41_w4a8_quantization.json").read_bytes())
        HookRegistry.reset()
        C.register_hooks()
        HookRegistry.apply_hooks()
        config = CompressedTensorsConfig.from_config(deepcopy(raw))
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        for mega in [False, True]:
            with patch.object(C, "is_hcu", return_value=True), patch.object(U, "get_moe_a2a_backend", return_value=S(is_megamoe=lambda: mega, is_deepep=lambda: not mega)), patch.object(U, "get_moe_runner_backend", return_value=S(is_deep_gemm=lambda: True)), patch("sglang.srt.runtime_context.get_disagg", return_value=S(disaggregation_mode="prefill")):
                method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
                self.assertIsInstance(method, W.HCUW4A8Int8DeepEPMoEMethod)
                self.assertIs(method.quant_config, config)
                self.assertEqual(method.use_mega_moe, mega)

    def test_mega_moe_rejects_decode(self):
        with patch("sglang.srt.runtime_context.get_disagg", return_value=S(disaggregation_mode="decode")):
            with self.assertRaisesRegex(ValueError, "prefill-only"):
                W.HCUW4A8Int8DeepEPMoEMethod(object(), use_mega_moe=True)

    def test_mega_postload_converts_canonical_weights_once(self):
        layer = torch.nn.Module()
        for prefix in ("w13", "w2"):
            layer.register_parameter(prefix + "_weight_packed", torch.nn.Parameter(torch.full((2, 2, 8), 0x12345678, dtype=torch.int32), requires_grad=False))
            layer.register_parameter(prefix + "_weight_scale", torch.nn.Parameter(torch.ones(2, 1, 8), requires_grad=False))
        with patch("sglang.srt.runtime_context.get_disagg", return_value=S(disaggregation_mode="prefill")):
            method = W.HCUW4A8Int8DeepEPMoEMethod(object(), use_mega_moe=True)
        def build(actual):
            self.assertIs(actual, layer)
            for prefix in ("w13", "w2"):
                self.assertFalse(hasattr(actual, prefix + "_weight_packed"))
                self.assertEqual(getattr(actual, prefix + "_weight").dtype, torch.int8)
                torch.testing.assert_close(getattr(actual, prefix + "_weight_scale"), torch.full((2, 8, 1), 1 / 16))
            actual._mega_moe_weights_built = True
        with patch("hcu_sglang.layers.moe.mega_moe.build_hcu_w4a8_mega_moe_experts_weights", side_effect=build) as builder:
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)
            self.assertEqual(builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
