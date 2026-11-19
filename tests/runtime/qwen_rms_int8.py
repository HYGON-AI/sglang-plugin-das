from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

from hcu_sglang.layers import qwen_rms_int8 as fusion
from hcu_sglang.layers.quantization import compressed_int8


class QwenRmsInt8Tests(TestCase):
    def setUp(self):
        self.x = torch.ones(2, 16, dtype=torch.bfloat16)
        self.weight = torch.ones_like(self.x[0])
        self.norm = SimpleNamespace(gemma_weight=self.weight, variance_epsilon=1e-6)
        self.scheme = object.__new__(fusion.CompressedTensorsW8A8Int8)
        self.scheme.is_static_input_scheme = False
        self.scheme.input_symmetric = True
        self.scheme.strategy = QuantizationStrategy.CHANNEL
        self.scheme.w8a8_strategy = 1
        self.attn = SimpleNamespace(scheme=self.scheme, weight=self.x, weight_scale=self.weight)
        self.ffn = SimpleNamespace(scheme=self.scheme)
        self.decoder = SimpleNamespace(
            qkv_proj=self.attn,
            mlp=SimpleNamespace(gate_up_proj=self.ffn),
            input_layernorm=self.norm,
            post_attention_layernorm=SimpleNamespace(),
        )
        self.quant = (self.x.to(torch.int8), torch.ones(2, 1))

    def test_gemm_consumes_fused_input_for_matching_projection(self):
        def forward(layer):
            out = fusion.try_fused_norm(layer.input_layernorm, self.x, None)
            self.assertIsNot(out, self.x)
            self.assertIsNone(fusion.take_prequantized_input(self.ffn, out))
            return compressed_int8.apply_weights(self.scheme, self.attn, out, None)

        with (
            patch.object(fusion, "supports_fused_rms_quant_input", return_value=True),
            patch("lightop.rms_norm_dynamic_per_token_quant_sglang", return_value=self.quant),
            patch.object(fusion, "enabled", True),
            patch.object(compressed_int8, "per_token_quant_int8") as separate_quant,
            patch.object(compressed_int8.ops, "triton_scaled_mm") as gemm,
        ):
            fusion.around_decoder_forward(forward, self.decoder)
        separate_quant.assert_not_called()
        self.assertIs(gemm.call_args.args[0], self.quant[0])
        self.assertIs(gemm.call_args.args[2], self.quant[1])
        self.assertIs(gemm.call_args.kwargs["out_dtype"], torch.bfloat16)

    def test_residual_update_and_fallback_on_different_input(self):
        def forward(layer):
            out, residual = fusion.try_fused_norm(layer.input_layernorm, self.x, r)
            self.assertIs(out, self.x)
            self.assertIs(residual, r)
            self.assertIsNone(fusion.take_prequantized_input(self.attn, out.clone()))

        r = self.x.clone()
        with (
            patch.object(fusion, "supports_fused_rms_quant_input", return_value=True),
            patch("lightop.rms_norm_dynamic_per_token_quant_sglang", return_value=self.quant) as rms,
        ):
            fusion.around_decoder_forward(forward, self.decoder)
        self.assertIs(rms.call_args.kwargs["residual"], r)
        self.assertTrue(rms.call_args.kwargs["update_input"])
        self.assertIsNone(fusion.take_prequantized_input(self.attn, self.x))

    def test_static_activation_falls_back(self):
        self.scheme.is_static_input_scheme = True

        def forward(layer):
            self.assertIsNone(fusion.try_fused_norm(layer.input_layernorm, self.x, None))

        with patch.object(fusion, "supports_fused_rms_quant_input") as probe:
            fusion.around_decoder_forward(forward, self.decoder)
        probe.assert_not_called()


if __name__ == "__main__":
    main()
