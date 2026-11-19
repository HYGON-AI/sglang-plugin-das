# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.environ import register_envs

register_envs()

from sglang.srt.environ import envs
from hcu_sglang.models import hyv4_env_hooks as hooks
from hcu_sglang.models.hyv4_gate_cp_shard import create_cp_sharded_gate_weight
from hcu_sglang.layers.attention.current_dsa.concat import concat_mla_absorb_q_general


class EnvironmentAliases(unittest.TestCase):
    def test_legacy_names_preserve_explicit_current_values(self):
        aliases = (
            ("SGLANG_NSA_FUSE_TOPK", "SGLANG_DSA_FUSE_TOPK", "1"),
            ("SGLANG_DSA_MQA_LOGITS_MEMORY_BUDGET_GB", "SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB", "2"),
            ("SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD", "0"),
            ("SGLANG_USE_FUSED_RMS_QUANT", "SGLANG_USE_LEGACY_FUSED_RMS_QUANT", "1"),
        )
        for old, current, value in aliases:
            with self.subTest(name=old), patch.dict(os.environ):
                os.environ.pop(current, None)
                os.environ[old] = value
                register_envs()
                self.assertEqual(os.environ[current], value)
                os.environ[current] = "0"
                register_envs()
                self.assertEqual(os.environ[current], "0")

    def test_disabled_ihc_keeps_current_callable(self):
        original = Mock(return_value="baseline")
        with envs.SGLANG_OPT_HY4_IHC_TILELANG.override(False):
            self.assertEqual(hooks.around_ihc_pre(original, None, None), "baseline")
            original.assert_called_once_with(None, None, None, 0.0)

    def test_concat_explicitly_disabled(self):
        nope = torch.randn(3, 2, 512)
        rope = torch.randn(3, 2, 64)
        with patch.object(hooks, "is_hcu", return_value=True):
            with envs.SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q.override(False):
                actual = concat_mla_absorb_q_general(nope, rope)
        torch.testing.assert_close(actual, torch.cat([nope, rope], -1))

    def test_cp_gate_uses_current_linear_partition_api(self):
        from hcu_sglang.layers import current_linear, linear_fusion
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        linear_fusion.register_hooks()
        HookRegistry.apply_hooks()
        group = SimpleNamespace(rank_in_group=0, world_size=8, ranks=list(range(8)))
        parallel = SimpleNamespace(
            enable_prefill_cp=True, attn_cp_size=8, attn_cp_rank=0,
            attn_tp_size=1, attn_dcp_size=1, tp_size=8, tp_rank=0,
            tp_group=group, attn_cp_group=group,
        )
        layer = SimpleNamespace(
            local_gate_width=64,
            linear_gate=SimpleNamespace(quant_method=UnquantizedLinearMethod()),
            _gate_fallback_backend="torch",
        )
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch.object(current_linear.N, "get_parallel", return_value=parallel),
            patch("hcu_sglang.models.hyv4_gate_cp_shard.get_gate_cp_shard_runtime"),
            patch.object(hooks, "is_hcu", return_value=True),
            envs.SGLANG_HY4_LINEAR_GATE_CP_SHARD.override(True),
        ):
            hooks.around_attention_init(
                lambda *args, **kwargs: None, layer,
                SimpleNamespace(hidden_size=32), 0,
            )
        self.assertEqual(tuple(layer.linear_gate.weight.shape), (8, 32))
        self.assertEqual(layer.linear_gate.parallel_group, "tp")

    def test_gate_row_shards_reconstruct_original_weight(self):
        weight = torch.arange(64 * 32).reshape(64, 32)
        shards = [create_cp_sharded_gate_weight(weight, 8, rank) for rank in range(8)]
        torch.testing.assert_close(torch.cat(shards), weight)
        for shard in shards:
            self.assertTrue(shard.is_contiguous())


@unittest.skipUnless(torch.cuda.is_available(), "requires HCU")
class HCUKernels(unittest.TestCase):
    @torch.inference_mode()
    def test_ihc_pre_post_head_with_norm(self):
        torch.manual_seed(17)
        device = "cuda"
        width = 6144
        for count in (0, 1, 17, 128):
            with self.subTest(rows=count):
                x = torch.randn(count, 4, width, device=device, dtype=torch.bfloat16)
                fn = torch.randn(8, 4 * width, device=device) / (4 * width) ** 0.5
                scale = torch.tensor([0.5, 0.5], device=device)
                base = torch.zeros(8, device=device)
                norm_weight = torch.ones(width, device=device, dtype=torch.bfloat16)
                layer = SimpleNamespace(
                    hc_fn=SimpleNamespace(weight=fn),
                    hc_scale=scale, hc_base=base,
                    rms_norm_eps=1e-6, hc_eps=1e-6, magnitude=2.0,
                )
                flat = x.flatten(1).float()
                inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
                gate = torch.nn.functional.linear(flat, fn) * inv
                pre = torch.sigmoid(gate[:, :4] * scale[0] + base[:4]) + 1e-6
                post = 2 * torch.sigmoid(gate[:, 4:] * scale[1] + base[4:]) + 1e-6
                reduced = (pre[:, :, None] * x.float()).sum(1).to(x.dtype)
                value = reduced.float()
                reference = (
                    value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    * norm_weight.float()
                ).to(x.dtype)
                original = Mock(side_effect=AssertionError("unexpected fallback"))
                with envs.SGLANG_OPT_HY4_IHC_TILELANG.override(True):
                    actual, actual_post = hooks.around_ihc_pre(
                        original, layer, x, norm_weight, 1e-6
                    )
                    torch.testing.assert_close(actual, reference, atol=0.035, rtol=0.02)
                    torch.testing.assert_close(actual_post, post, atol=2e-4, rtol=2e-4)
                    branch = torch.randn(count, width, device=device, dtype=x.dtype)
                    updated = hooks.around_ihc_post(original, layer, branch, x, actual_post)
                    expected = (actual_post[:, :, None] * branch.float()[:, None] + x.float()).to(x.dtype)
                    torch.testing.assert_close(updated, expected, atol=0.035, rtol=0.02)

                    head_fn = fn[:4].contiguous()
                    head = SimpleNamespace(
                        hc_head_fn=SimpleNamespace(weight=head_fn),
                        hc_head_scale=scale[:1], hc_head_base=base[:4],
                        config=SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-6),
                    )
                    head_gate = torch.sigmoid(
                        torch.nn.functional.linear(flat, head_fn) * inv * scale[:1] + base[:4]
                    ) + 1e-6
                    expected = (head_gate[:, :, None] * x.float()).sum(1).to(x.dtype)
                    output = hooks.around_ihc_head(original, head, x)
                    torch.testing.assert_close(output, expected, atol=0.035, rtol=0.02)

    @torch.inference_mode()
    def test_lightop_prefill_dequant_matches_packed_layout(self):
        from hcu_sglang.layers.attention.current_dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )

        cache = torch.zeros(64, 656, device="cuda", dtype=torch.uint8)
        values = torch.randn(64, 512, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(64, 4, device="cuda") + 0.1
        rope = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        cache[:, :512] = values.view(torch.uint8)
        cache[:, 512:528] = scales.view(torch.uint8)
        cache[:, 528:] = rope.view(torch.uint8)
        indices = torch.tensor([7, 3, 15, 0, 55, 55], device="cuda", dtype=torch.int32)
        full = torch.cat([
            (values.float().view(64, 4, 128) * scales[:, :, None]).view(64, 512),
            rope.float(),
        ], -1).to(torch.bfloat16)
        expected = full[indices.long()][:, None, :]
        with envs.SGLANG_DSA_LIGHTOP_PREFILL_DEQUANT_KV.override(True):
            actual = dequantize_k_cache_paged(cache.view(torch.float8_e4m3fn), indices)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    @torch.inference_mode()
    def test_concat_enabled_matches_torch(self):
        nope = torch.randn(17, 8, 512, device="cuda", dtype=torch.bfloat16)
        rope = torch.randn(17, 8, 64, device="cuda", dtype=torch.bfloat16)
        with envs.SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q.override(True):
            actual = concat_mla_absorb_q_general(nope, rope)
        torch.testing.assert_close(actual, torch.cat([nope, rope], -1), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
