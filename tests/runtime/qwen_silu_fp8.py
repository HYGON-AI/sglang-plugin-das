# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, PropertyMock, patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
from hcu_sglang.environ import register_envs

bootstrap_aiter_compat()
register_envs()

from hcu_sglang.models.qwen import silu_fp8


class QwenSiluFp8Tests(TestCase):
    def setUp(self):
        self.x = torch.ones((2, 5120), dtype=torch.bfloat16)
        self.gate_up = torch.ones((2, 34816), dtype=torch.bfloat16)
        self.output = object()
        self.proj = Mock(
            weight=SimpleNamespace(dtype=torch.float8_e4m3fn),
            input_is_parallel=True,
            input_size_per_partition=17408,
            return_value=(self.output, None),
        )
        self.proj.supports_fused_silu_mul_fp8_quant_input.return_value = True
        self.layer = SimpleNamespace(
            down_proj=self.proj,
            gate_up_proj=Mock(return_value=(self.gate_up, None)),
            act_fn=Mock(),
            _enable_silu_fp4_quant_fusion=False,
        )
        self.original = Mock(return_value=self.output)
        ctx = ExitStack()
        self.addCleanup(ctx.close)
        ctx.enter_context(
            patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True)
        )
        ctx.enter_context(patch.object(silu_fp8, "_is_gfx938", return_value=True))
        ctx.enter_context(
            patch.object(silu_fp8, "get_forward", return_value=SimpleNamespace(sp_active=False))
        )
        exec_context = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
        ctx.enter_context(patch.object(silu_fp8, "get_exec", return_value=exec_context))

    def test_registers_only_with_the_existing_fp8_switch(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(silu_fp8, "is_hcu", return_value=True),
                patch.dict(os.environ, {"SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT": str(int(enabled))}),
                patch.object(silu_fp8.HookRegistry, "register") as register,
            ):
                silu_fp8.register()
                if enabled:
                    register.assert_called_once_with(
                        "sglang.srt.models.qwen2_moe.Qwen2MoeMLP.forward",
                        silu_fp8.around_forward,
                        silu_fp8.HookType.AROUND,
                    )
                else:
                    register.assert_not_called()

    def test_passes_packed_gate_up_to_the_fused_row_projection(self):
        result = silu_fp8.around_forward(self.original, self.layer, self.x)
        self.assertIs(result, self.output)
        self.layer.gate_up_proj.assert_called_once_with(self.x)
        self.proj.assert_called_once_with(self.gate_up, use_fused_silu_mul_fp8_quant=True)
        self.layer.act_fn.assert_not_called()
        self.original.assert_not_called()

    def test_incompatible_gate_layout_reuses_produced_activation(self):
        strided = torch.empty((2, 69632), dtype=torch.bfloat16)[:, ::2]
        self.layer.gate_up_proj.return_value = (strided, None)
        result = silu_fp8.around_forward(self.original, self.layer, self.x)
        self.assertIs(result, self.output)
        self.layer.gate_up_proj.assert_called_once_with(self.x)
        self.layer.act_fn.assert_called_once_with(strided)
        self.proj.assert_called_once_with(self.layer.act_fn.return_value)
        self.original.assert_not_called()

    def test_unsupported_down_projection_delegates_before_gate_up(self):
        self.proj.supports_fused_silu_mul_fp8_quant_input.return_value = False
        result = silu_fp8.around_forward(self.original, self.layer, self.x)
        self.assertIs(result, self.output)
        self.original.assert_called_once_with(self.layer, self.x)
        self.layer.gate_up_proj.assert_not_called()


if __name__ == "__main__":
    main()
