from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import hcu_sglang
from sglang.srt.plugins.hook_registry import HookRegistry


class QuantizationMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hcu_sglang.register()
        HookRegistry.apply_hooks()
        from sglang.srt.layers.quantization import auto_round, kv_cache, moe_wna16
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        cls.kv_cache = kv_cache.BaseKVCacheMethod
        cls.moe_wna16 = moe_wna16.MoeWNA16Config
        cls.auto_round = auto_round.AutoRoundConfig
        cls.fused_moe = FusedMoE

    def test_kv_cache_requires_checkpoint_scales_and_preserves_default(self):
        layer = SimpleNamespace(
            k_scale=torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False),
            v_scale=torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False),
            layer_id=7,
        )
        strict = self.kv_cache(
            SimpleNamespace(kv_cache_scheme={"require_checkpoint_scales": True})
        )
        with self.assertRaisesRegex(RuntimeError, "layer 7"):
            strict.process_weights_after_loading(layer)
        default = self.kv_cache(SimpleNamespace(kv_cache_scheme={}))
        default.process_weights_after_loading(layer)
        self.assertEqual(layer.k_scale_float, 1.0)
        self.assertEqual(layer.v_scale_float, 1.0)

    def test_moe_wna16_hcu_skips_gpu_capability_and_non_hcu_keeps_rejection(self):
        value = self.moe_wna16("awq", 4, 32, False, False, None, {})
        self.assertFalse(value.use_marlin)
        from hcu_sglang.layers.quantization import moe_wna16 as owned

        with (
            patch.object(owned, "_is_hcu", False),
            patch.object(owned, "get_device_capability", return_value=(0, 0)),
        ):
            with self.assertRaisesRegex(ValueError, "not supported"):
                owned.moe_wna16_init(
                    object.__new__(self.moe_wna16),
                    "awq",
                    4,
                    32,
                    False,
                    False,
                    None,
                    {},
                )

    def test_auto_round_recognizes_actual_fused_moe_not_only_class_name(self):
        config = object.__new__(self.auto_round)
        config.extra_config = {
            "model.layers.0.experts": {
                "bits": 4,
                "group_size": 32,
                "sym": True,
            }
        }
        config.block_name_to_quantize = None
        config.packed_modules_mapping = {}
        config.weight_bits = 8
        config.group_size = 64
        config.sym = False
        layer = object.__new__(self.fused_moe)
        actual = config.get_layer_config(layer, "model.layers.0.experts")
        self.assertEqual(actual, (4, 32, True))


if __name__ == "__main__":
    unittest.main()
