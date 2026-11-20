# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from glm52_cpu_support import isolated_runtime

class GlmSlimQuantCpuTests(unittest.TestCase):
    def make(self, rt, quant, *, glm=True, hashed=False, v4=False):
        config = SimpleNamespace(model_type='glm_moe_dsa' if glm else 'deepseek_v3', n_routed_experts=16, hidden_size=64, topk_method='noaux_tc', router_fp32=False)
        q = None if quant is None else SimpleNamespace(get_name=lambda: quant)
        old = rt.torch.get_default_dtype()
        try:
            rt.torch.set_default_dtype(rt.torch.bfloat16)
            gate = rt.native.MoEGate(config, q, is_hash_moe=hashed, is_deepseek_v4=v4)
        finally:
            rt.torch.set_default_dtype(old)
        return config, q, gate

    def test_constructor_scope_and_disabled_path(self):
        with isolated_runtime() as rt:
            cases = itertools.product([False, True], [False, True], [None, 'compressed_tensors', 'slimquant_w4a8', 'slimquant_w4a8_marlin', 'fp8'], [False, True], [False, True])
            for enabled, glm, quant, hashed, v4 in cases:
                with self.subTest(enabled=enabled, glm=glm, quant=quant, hashed=hashed, v4=v4):
                    rt.flag.value = enabled
                    _, _, gate = self.make(rt, quant, glm=glm, hashed=hashed, v4=v4)
                    expected = enabled and glm and quant in ('compressed_tensors', 'slimquant_w4a8', 'slimquant_w4a8_marlin') and not hashed and not v4
                    self.assertEqual(gate._hcu_bf16_router, expected)

    def test_native_bf16_forward_keeps_dtype_and_values(self):
        with isolated_runtime() as rt:
            torch = rt.torch
            rt.native.tiny_gemm_bf16 = Mock(side_effect=AssertionError('FP32 router must not run'))
            cases = itertools.product(['compressed_tensors', 'slimquant_w4a8', 'slimquant_w4a8_marlin'], [1, 4, 8, 16], [torch.bfloat16, torch.float32])
            for quant, rows, dtype in cases:
                with self.subTest(quant=quant, rows=rows, dtype=dtype):
                    _, _, gate = self.make(rt, quant)
                    with torch.no_grad():
                        gate.weight.copy_(torch.arange(gate.weight.numel()).reshape_as(gate.weight).to(gate.weight.dtype) / 1024)
                    x = torch.linspace(-1, 1, rows * 64).reshape(rows, 64).to(dtype)
                    out = gate(x)
                    ref = torch.nn.functional.linear(x.to(gate.weight.dtype), gate.weight)
                    self.assertEqual(out.dtype, torch.bfloat16)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
            rt.native.tiny_gemm_bf16.assert_not_called()

    def test_env_off_preserves_existing_fp32_path(self):
        with isolated_runtime() as rt:
            rt.flag.value = False
            _, _, gate = self.make(rt, 'slimquant_w4a8_marlin')
            with rt.torch.no_grad():
                gate.weight.fill_(0.125)
            out = gate(rt.torch.ones(1, 64, dtype=rt.torch.bfloat16))
            self.assertEqual(out.dtype, rt.torch.float32)

    def test_eh_norm_quantization_and_platform_scope(self):
        with isolated_runtime() as rt:
            for hcu, glm, quant in itertools.product([False, True], [False, True], [None, 'compressed_tensors', 'slimquant_w4a8', 'slimquant_w4a8_marlin', 'fp8']):
                with self.subTest(hcu=hcu, glm=glm, quant=quant):
                    rt.compat.is_hcu = lambda: hcu
                    config = SimpleNamespace(model_type='glm_moe_dsa' if glm else 'deepseek_v3')
                    q = None if quant is None else SimpleNamespace(get_name=lambda: quant)
                    self.assertEqual(rt.nextn._use_native_eh_norm(config, q), hcu and glm and quant in ('compressed_tensors', 'slimquant_w4a8', 'slimquant_w4a8_marlin'))

    def test_non_hcu_hooks_preserve_effective_original(self):
        with isolated_runtime() as rt:
            rt.gate.is_hcu = lambda: False
            for hook in [rt.gate.around_init, rt.gate.around_forward]:
                original = Mock(return_value=object())
                receiver, value = object(), object()
                self.assertIs(hook(original, receiver, value, named=3), original.return_value)
                original.assert_called_once_with(receiver, value, named=3)

if __name__ == '__main__':
    unittest.main()
