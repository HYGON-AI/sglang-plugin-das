# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
from hcu_sglang.environ import register_envs

bootstrap_aiter_compat()
register_envs()

from compressed_tensors.quantization import QuantizationArgs
from sglang.srt.plugins.hook_registry import HookRegistry

from hcu_sglang.layers import qwen_rms_fp8 as fusion
from hcu_sglang.layers.quantization import compressed_fp8
from hcu_sglang.models.qwen import qwen3_5


class QwenRmsFp8Tests(TestCase):
    def setUp(self):
        ctx = ExitStack()
        self.addCleanup(ctx.close)
        ctx.enter_context(patch.object(fusion, "get_parallel", return_value=SimpleNamespace(tp_size=1)))
        ctx.enter_context(patch.object(fusion, "get_forward", return_value=SimpleNamespace(sp_active=False)))
        ctx.enter_context(patch.object(fusion, "enabled", True))
        ctx.enter_context(patch.object(compressed_fp8, "_is_hcu", True))
        ctx.enter_context(patch.object(compressed_fp8, "_use_lightop_channel_fp8", False))
        ctx.enter_context(patch.object(compressed_fp8.Q, "_use_aiter", False))
        ctx.enter_context(
            patch.object(
                fusion.CompressedTensorsW8A8Fp8,
                "supports_fp8_prequantized_input",
                compressed_fp8._PREQUANTIZED_CAPABILITY,
                create=True,
            )
        )
        scheme = fusion.CompressedTensorsW8A8Fp8(
            QuantizationArgs(num_bits=8, type="float", strategy="channel"), False
        )
        self.proj = SimpleNamespace(
            scheme=scheme,
            weight=torch.ones(5120, 4).to(torch.float8_e4m3fn),
            weight_scale=torch.ones(4, 1),
        )
        self.x = torch.ones(2, 5120, dtype=torch.bfloat16)
        self.residual = torch.full_like(self.x, 2)
        self.weight = torch.zeros_like(self.x[0])
        self.quant = (self.x.to(torch.float8_e4m3fn), torch.ones(2, 1))
        self.norm = SimpleNamespace(weight=self.weight, variance_epsilon=1e-6)
        self.decoder = SimpleNamespace(
            linear_attn=SimpleNamespace(in_proj_qkvz=self.proj),
            input_layernorm=self.norm,
            post_attention_layernorm=SimpleNamespace(),
            mlp=SimpleNamespace(),
        )
        self.kernel = Mock(side_effect=self.fused)
        ctx.enter_context(patch.object(fusion, "_kernel", return_value=self.kernel))
        ctx.enter_context(patch.object(fusion, "_supported_layout", return_value=True))

    def fused(self, x, weight, eps, *, fp8type, residual, update_input):
        self.assertIs(x, self.x)
        self.assertEqual(weight.data_ptr(), self.weight.data_ptr())
        self.assertEqual(eps, self.norm.variance_epsilon)
        self.assertEqual(fp8type, 0)
        self.assertIs(residual, self.residual)
        self.assertTrue(update_input)
        residual.add_(x)
        x.fill_(4)
        return self.quant

    def test_gdn_keeps_bf16_and_feeds_actual_fp8_consumer(self):
        gemm_result = torch.ones(2, 4, dtype=torch.bfloat16)

        def forward(layer):
            x, r = fusion.try_fused_norm(layer.input_layernorm, self.x, self.residual)
            self.assertIs(x, self.x)
            self.assertIs(r, self.residual)
            torch.testing.assert_close(x, torch.full_like(x, 4))
            torch.testing.assert_close(r, torch.full_like(r, 3))
            self.assertIsNone(fusion.take_prequantized_input(object(), x))
            output = compressed_fp8.apply_weights(self.proj.scheme, self.proj, x)
            self.assertIsNone(fusion.take_prequantized_input(self.proj, x))
            return output

        with patch.object(compressed_fp8.Q, "apply_fp8_linear", return_value=gemm_result) as gemm:
            self.assertIs(fusion.around_decoder_forward(forward, self.decoder), gemm_result)

        q, scale = gemm.call_args.kwargs["input"]
        self.assertIs(q, self.quant[0])
        self.assertIs(scale, self.quant[1])
        self.assertEqual(gemm.call_args.kwargs["pre_quant_output_dtype"], torch.bfloat16)
        self.kernel.assert_called_once()

    def test_context_cleanup_and_unmatched_input(self):
        def forward(layer):
            fusion.try_fused_norm(layer.input_layernorm, self.x, self.residual)
            self.assertIsNone(fusion.take_prequantized_input(self.proj, self.x.clone()))
            raise RuntimeError("decoder failed")

        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            fusion.around_decoder_forward(forward, self.decoder)
        self.assertIsNone(fusion._inputs.get())

    def test_merged_projection_avoids_unused_fusion(self):
        self.decoder.linear_attn._fused_in_proj_weight = torch.empty(0)

        def forward(layer):
            self.assertIsNone(fusion.try_fused_norm(layer.input_layernorm, self.x, self.residual))

        fusion.around_decoder_forward(forward, self.decoder)
        self.kernel.assert_not_called()

    def test_unsupported_layout_uses_original_norm(self):
        def forward(layer):
            self.assertIsNone(fusion.try_fused_norm(layer.input_layernorm, self.x, None))
            self.assertIsNone(fusion.take_prequantized_input(self.proj, self.x))

        with patch.object(fusion, "_supported_layout", return_value=False):
            fusion.around_decoder_forward(forward, self.decoder)
        self.kernel.assert_not_called()

    def test_registers_both_decoder_hooks(self):
        with patch.object(HookRegistry, "register") as register:
            qwen3_5.register()

        for cls in ("Qwen3_5AttentionDecoderLayer", "Qwen3_5LinearDecoderLayer"):
            register.assert_any_call(
                f"sglang.srt.models.qwen3_5.{cls}.forward",
                fusion.around_decoder_forward,
                qwen3_5.HookType.AROUND,
            )


if __name__ == "__main__":
    main()
