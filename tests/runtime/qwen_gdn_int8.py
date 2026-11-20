# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from hcu_sglang.layers.quantization import compressed_int8
from hcu_sglang.models.qwen import gdn_int8, qwen3_5
from hcu_sglang.models.qwen.gdn_int8 import GdnGatedInt8Input, _supports_fusion


class QwenGdnInt8Tests(TestCase):
    def setUp(self):
        self.scheme = object.__new__(compressed_int8.N.CompressedTensorsW8A8Int8)
        self.scheme.is_static_input_scheme = False
        self.scheme.input_symmetric = True
        self.scheme.w8a8_strategy = 1
        self.projection = SimpleNamespace(
            scheme=self.scheme, input_is_parallel=True,
            weight=torch.empty((128, 64), dtype=torch.int8),
            weight_scale=torch.empty((64, 1), dtype=torch.float32),
        )

    def test_prequantized_projection_bypasses_standalone_int8_quant(self):
        q = torch.empty((4, 128), dtype=torch.int8)
        scale = torch.ones((4, 1))
        inp = GdnGatedInt8Input(q, scale, torch.bfloat16)
        with (
            patch.object(compressed_int8, "_use_fused_rms_quant", False),
            patch.object(compressed_int8, "per_token_quant_int8") as quant,
            patch.object(compressed_int8.ops, "triton_scaled_mm") as gemm,
        ):
            compressed_int8.apply_weights(self.scheme, self.projection, inp, None)
        quant.assert_not_called()
        self.assertIs(gemm.call_args.args[0], q)
        self.assertIs(gemm.call_args.args[2], scale)
        self.assertIs(gemm.call_args.kwargs["out_dtype"], torch.bfloat16)

    def test_registers_gdn_forward_only_when_enabled(self):
        target = "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward"
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(gdn_int8, "enabled", enabled),
                patch.object(qwen3_5.HookRegistry, "register") as register,
            ):
                qwen3_5.register()
                hooks = [
                    call.args for call in register.call_args_list if call.args[0] == target
                ]
                expected = [(target, gdn_int8.around_forward, qwen3_5.HookType.AROUND)] if enabled else []
                self.assertEqual(hooks, expected)

    def test_requires_dynamic_symmetric_local_projection(self):
        layer = SimpleNamespace(
            out_proj=self.projection, head_v_dim=128,
            norm=SimpleNamespace(
                norm_before_gate=True, group_size=None, activation="swish", bias=None,
            ),
        )
        self.assertTrue(_supports_fusion(layer))
        self.scheme.is_static_input_scheme = True
        self.assertFalse(_supports_fusion(layer))
        self.scheme.is_static_input_scheme = False
        self.projection.input_is_parallel = False
        self.assertFalse(_supports_fusion(layer))


if __name__ == "__main__":
    main()
