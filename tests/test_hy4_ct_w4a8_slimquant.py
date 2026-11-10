# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 compressed-tensors W4A8 under slimquant_w4a8_marlin; runs in the HCU container."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from unittest.mock import Mock

_HAVE_RUNTIME = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "sglang")
)

_INT4_ACT = {
    "num_bits": 8,
    "type": "int",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}


def _group(bits, targets, fmt=None):
    group = {
        "weights": {
            "num_bits": bits,
            "type": "int",
            "strategy": "channel",
            "symmetric": True,
            "dynamic": False,
            "group_size": None,
            "actorder": None,
        },
        "input_activations": dict(_INT4_ACT),
        "output_activations": None,
        "targets": targets,
    }
    if fmt is not None:
        group["format"] = fmt
    return group


_EXPERTS = r"re:.*layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
_MLA = r"re:.*self_attn\.(?:fused_qkv_a_proj(?:_with_mqa)?|q_b_proj|kv_b_proj|o_proj)$"
_SHARED = r"re:.*shared_experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj)$"


def _hy4_mixed():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": [r"re:.*(?:router\.)?gate$", "lm_head", r"re:.*indexer\.weights_proj$"],
        "config_groups": {
            "group_int4_routed_experts": _group(4, [_EXPERTS], "pack-quantized"),
            "group_int8_mla": _group(8, [_MLA], "int-quantized"),
            "group_int8_mlp": _group(8, [_SHARED], "int-quantized"),
        },
    }


def _all_w4():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "config_groups": {"group_0": _group(4, [_EXPERTS])},
    }


@unittest.skipUnless(_HAVE_RUNTIME, "requires torch + official sglang")
class HY4CompressedConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as m

        cls.m = m
        cls.Config = m.SlimQuantW4A8Int8MarlinConfig

    def _from_config(self, cfg, arch):
        cfg = copy.deepcopy(cfg)
        cfg["hf_config"] = {"architectures": [arch]}
        cfg["packed_modules_mapping"] = {"gate_up_proj": ["gate_proj", "up_proj"]}
        return self.Config.from_config(cfg)

    def test_hy4_mixed_int4_int8_is_accepted(self):
        cfg = _hy4_mixed()
        self.assertEqual(
            self.Config.override_quantization_method(cfg, "slimquant_w4a8_marlin"),
            "slimquant_w4a8_marlin",
        )
        qc = self._from_config(cfg, "HYV4ForCausalLM")
        self.assertEqual(qc.checkpoint_format, self.m.HY4_CT_W4A8_FORMAT)
        self.assertEqual(qc.linear_targets, [_MLA, _SHARED])

    def test_all_w4_non_hyv4_keeps_existing_path(self):
        cfg = _all_w4()
        self.assertEqual(
            self.Config.override_quantization_method(cfg, "slimquant_w4a8_marlin"),
            "slimquant_w4a8_marlin",
        )
        qc = self._from_config(cfg, "DeepseekV3ForCausalLM")
        self.assertIsNone(qc.checkpoint_format)
        self.assertIsNone(qc.linear_targets)
        self.assertTrue(qc.compressed_tensors_packed)

    def test_int8_groups_require_hyv4(self):
        cfg = _hy4_mixed()
        with self.assertRaisesRegex(ValueError, "HYV4"):
            self._from_config(cfg, "DeepseekV3ForCausalLM")

    def test_group_violations_are_rejected(self):
        bad_format = _hy4_mixed()
        bad_format["config_groups"]["group_int8_mla"]["format"] = "pack-quantized"
        bad_act = _hy4_mixed()
        bad_act["config_groups"]["group_int4_routed_experts"]["input_activations"] = None
        only_int8 = _hy4_mixed()
        del only_int8["config_groups"]["group_int4_routed_experts"]
        bad_strategy = _hy4_mixed()
        bad_strategy["config_groups"]["group_int8_mlp"]["weights"]["strategy"] = "group"
        for cfg in (bad_format, bad_act, only_int8, bad_strategy):
            self.assertIsNone(
                self.Config.override_quantization_method(cfg, "slimquant_w4a8_marlin")
            )
            with self.assertRaises(ValueError):
                self._from_config(cfg, "HYV4ForCausalLM")
        self.assertIsNone(
            self.Config.override_quantization_method(_hy4_mixed(), "slimquant_marlin")
        )

    def test_registered_lazy_config_forwards_compressed_tensors(self):
        from hcu_sglang.layers.quantization.registry import (
            LazySlimQuantW4A8Int8MarlinConfig as Lazy,
        )

        self.assertEqual(
            Lazy.override_quantization_method(_hy4_mixed(), "slimquant_w4a8_marlin"),
            "slimquant_w4a8_marlin",
        )
        self.assertIsNone(
            Lazy.override_quantization_method(_hy4_mixed(), "w4a8_int8")
        )
        self.assertIsNone(Lazy.override_quantization_method(_hy4_mixed(), None))

    def test_linear_targets_select_w8a8_or_bf16(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from hcu_sglang.layers.quantization.slimquant_w4a8 import (
            SlimQuantW4A8Int8LinearMethod,
        )

        qc = self._from_config(_hy4_mixed(), "HYV4ForCausalLM")
        layer = Mock(spec=LinearBase)
        cases = {
            "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa": SlimQuantW4A8Int8LinearMethod,
            "model.layers.3.self_attn.o_proj": SlimQuantW4A8Int8LinearMethod,
            "model.layers.3.mlp.shared_experts.gate_up_proj": SlimQuantW4A8Int8LinearMethod,
            "model.layers.3.mlp.shared_experts.down_proj": SlimQuantW4A8Int8LinearMethod,
            "model.layers.3.self_attn.indexer.weights_proj": UnquantizedLinearMethod,
            "model.layers.3.self_attn.indexer.wq_b": UnquantizedLinearMethod,
            "lm_head": UnquantizedLinearMethod,
        }
        for prefix, expected in cases.items():
            with self.subTest(prefix=prefix):
                self.assertIsInstance(qc.get_quant_method(layer, prefix), expected)


@unittest.skipUnless(_HAVE_RUNTIME, "requires torch + official sglang")
class HY4NormalizeCheckpointWeightsTests(unittest.TestCase):
    def test_ct_pack_to_legacy_round_trip(self):
        import torch
        from torch import nn
        from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as m

        gen = torch.Generator().manual_seed(0)
        E, N, K = 2, 4, 16
        q = torch.randint(-8, 8, (E, N, K), generator=gen, dtype=torch.int32)

        u = (q + 8).view(E, N, K // 8, 8)
        ct = torch.zeros(E, N, K // 8, dtype=torch.int32)
        for i in range(8):
            ct |= u[..., i] << (4 * i)
        ct_int8 = ct.contiguous().view(torch.int8).view(E, N, K // 2)

        lo = (q[..., 0::2] & 0xF) << 4
        hi = q[..., 1::2] & 0xF
        legacy = (lo | hi).to(torch.uint8).view(torch.int8)

        scale = torch.rand(E, N, 1, generator=gen) + 0.5
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(ct_int8.clone(), requires_grad=False)
        layer.w2_weight = nn.Parameter(ct_int8.clone(), requires_grad=False)
        layer.w13_weight_scale = nn.Parameter(scale.clone(), requires_grad=False)
        layer.w2_weight_scale = nn.Parameter(scale.clone(), requires_grad=False)

        qc = m.SlimQuantW4A8Int8MarlinConfig(checkpoint_format=m.HY4_CT_W4A8_FORMAT)
        qc.normalize_checkpoint_weights(layer)
        for name in ("w13", "w2"):
            self.assertTrue(torch.equal(getattr(layer, f"{name}_weight").data, legacy))
            self.assertTrue(
                torch.allclose(getattr(layer, f"{name}_weight_scale").data, scale / 16)
            )

        layer.w13_weight.data.copy_(ct_int8)
        m.SlimQuantW4A8Int8MarlinConfig().normalize_checkpoint_weights(layer)
        self.assertTrue(torch.equal(layer.w13_weight.data, ct_int8))


@unittest.skipUnless(_HAVE_RUNTIME, "requires torch + official sglang")
class HYV4PackedNameAdapterTests(unittest.TestCase):
    def test_packed_expert_names(self):
        import torch
        from hcu_sglang.models.hunyuan_v4 import adapt_hyv4_packed_weight

        qc = Mock(checkpoint_format="hy4_compressed_tensors_w4a8")
        base = "model.layers.5.mlp.experts.7.gate_proj"
        packed = torch.arange(8, dtype=torch.int32).view(2, 4)

        self.assertEqual(
            adapt_hyv4_packed_weight(f"{base}.weight_shape", torch.tensor([2, 32]), qc),
            (None, None),
        )
        name, weight = adapt_hyv4_packed_weight(f"{base}.weight_packed", packed, qc)
        self.assertEqual(name, f"{base}.weight")
        self.assertEqual(weight.dtype, torch.int8)
        self.assertEqual(tuple(weight.shape), (2, 16))
        self.assertTrue(torch.equal(weight.view(torch.int32), packed))

        scale = torch.ones(2, 1)
        self.assertIs(adapt_hyv4_packed_weight(f"{base}.weight_scale", scale, qc)[1], scale)

        for bad_name, bad in (
            ("model.layers.5.self_attn.o_proj.weight_packed", packed),
            (f"{base}.weight_packed", packed.to(torch.int8)),
            (f"{base}.weight_packed", packed.view(-1)),
        ):
            with self.assertRaises(ValueError):
                adapt_hyv4_packed_weight(bad_name, bad, qc)

        other = Mock(checkpoint_format=None)
        self.assertEqual(
            adapt_hyv4_packed_weight(f"{base}.weight_packed", packed, other),
            (f"{base}.weight_packed", packed),
        )


if __name__ == "__main__":
    unittest.main()
