# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.quantization import compressed_int8
from hcu_sglang.models.qwen import attention_int8, qwen3_5


class QwenAttentionInt8Tests(TestCase):
    def test_disabled_hook_delegates_to_original(self):
        original = Mock()
        arguments = tuple(object() for _ in range(4))
        with patch.object(attention_int8, "enabled", False):
            self.assertIs(qwen3_5.replacement_self_attention(original, *arguments), original.return_value)
        original.assert_called_once_with(*arguments)

    def test_projection_uses_int8_and_scale_directly(self):
        q = torch.zeros((1, 32), dtype=torch.int8)
        scale = torch.ones((1, 1))
        x = compressed_int8.PrequantizedInt8Input(q, scale, torch.bfloat16)
        layer = SimpleNamespace(weight=torch.zeros((32, 8)), weight_scale=torch.ones((8, 1)))
        with (
            patch.object(compressed_int8.ops, "triton_scaled_mm") as gemm,
            patch.object(compressed_int8, "per_token_quant_int8") as quant,
        ):
            compressed_int8.apply_weights(SimpleNamespace(w8a8_strategy=1), layer, x, None)
        quant.assert_not_called()
        self.assertIs(gemm.call_args.args[0], q)
        self.assertIs(gemm.call_args.args[2], scale)
        self.assertIs(gemm.call_args.kwargs["out_dtype"], torch.bfloat16)

    @skipUnless(torch.cuda.is_available(), "GPU required")
    def test_fused_path_calls_attention_once_and_preserves_fallback(self):
        attn = torch.ones((4, 3072), device="cuda", dtype=torch.bfloat16)
        gate = torch.zeros((4, 12, 512), device="cuda", dtype=attn.dtype)[..., 256:]
        scheme = object.__new__(attention_int8.CompressedTensorsW8A8Int8)
        scheme.is_static_input_scheme = False
        scheme.input_symmetric = True
        result = object()
        proj = Mock(scheme=scheme, input_is_parallel=True, input_size_per_partition=3072,
                    return_value=(result, None))
        layer = SimpleNamespace(
            o_proj=proj, attn_output_gate=True,
            _prepare_qkv_gate=Mock(return_value=(None, None, None, gate)),
            attn=Mock(return_value=attn),
        )
        original = Mock(return_value=result)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(name="TARGET_VERIFY"))
        with (
            patch.object(attention_int8, "enabled", True),
            patch.object(attention_int8, "get_forward", return_value=SimpleNamespace(sp_active=False)),
        ):
            self.assertIs(qwen3_5.replacement_self_attention(original, layer, None, attn, batch), result)
            original.assert_not_called()
            layer.attn.assert_called_once()
            x = proj.call_args.args[0]
            self.assertIsInstance(x, compressed_int8.PrequantizedInt8Input)
            self.assertEqual(tuple(x.activation.shape), (4, 3072))
            self.assertEqual(tuple(x.scale.shape), (4, 1))
            scheme.is_static_input_scheme = True
            self.assertIs(qwen3_5.replacement_self_attention(original, layer, None, attn, batch), result)
            original.assert_called_once_with(layer, None, attn, batch)
            layer.attn.assert_called_once()


if __name__ == "__main__":
    main()
