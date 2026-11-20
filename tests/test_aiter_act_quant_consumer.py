import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "hcu_sglang/layers/quantization/aiter_act_quant.py"
SPEC = importlib.util.spec_from_file_location("aiter_act_quant_test_target", PATH)
TARGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGET)

CORE = """
def core(sizes, dtype, activation='silu', use_int8_w4a8=True, block_shape=None,
         curr_a2_scale=None, is_gated=True, gemm1_alpha=None, gemm1_limit=None, N=1280):
    intermediate_cache1 = Tensor(dtype)
    intermediate_cache2 = object()
    for tokens_in_chunk in sizes:
        if (activation == 'silu' and is_gated
                and gemm1_alpha is None and gemm1_limit is None):
            moe_c_silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
        else:
            fallback_activation(intermediate_cache2, intermediate_cache1)
        invoke_fused_moe_kernel_marlin(intermediate_cache2, None, None,
                                      curr_a2_scale, is_bottom=True)
"""


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def view(self, *shape):
        return self


class ConsumerTests(unittest.TestCase):
    def test_private_namespace_is_built_after_runtime_initialization(self):
        state = []
        def original(hidden):
            return 'original'
        wrapped = TARGET.lazy_aiter_act_quant(original)
        with patch.object(TARGET, 'build_aiter_act_quant', side_effect=lambda fn: lambda hidden: tuple(state)) as build:
            self.assertEqual(wrapped(SimpleNamespace(shape=(4, 2560))), 'original')
            build.assert_not_called()
            state.append('expert-map-and-config-patches-ready')
            self.assertEqual(wrapped(SimpleNamespace(shape=(4096, 2560))), tuple(state))

    def run_core(self, sizes, **kwargs):
        calls = []
        triton_quant = kwargs.pop('triton_quant', True)
        q, scale = object(), object()
        def fuse(x, dtype):
            calls.append(('fused', x.dtype, dtype))
            return q, scale
        def consume(a, b, c, a_scale, **options):
            calls.append(('gemm2', a is q, a_scale is scale, a_scale))
        scope = dict(Tensor=Tensor, torch=SimpleNamespace(bfloat16='bf16', int8='int8'),
                     _hcu_use_triton_quant=lambda activation: triton_quant,
                     _hcu_fuse=fuse, invoke_fused_moe_kernel_marlin=consume,
                     moe_c_silu_and_mul=lambda *a: calls.append(('silu',)),
                     fallback_activation=lambda *a: calls.append(('fallback',)))
        transform = TARGET._FuseActivation()
        tree = transform.visit(ast.parse(CORE))
        self.assertEqual((transform.activations, transform.consumers), (1, 1))
        exec(compile(ast.fix_missing_locations(tree), '<consumer>', 'exec'), scope)
        scope['core'](sizes, **dict(dict(dtype='bf16'), **kwargs))
        return calls

    def test_down_projection_consumes_fused_codes_and_scales(self):
        calls = self.run_core([16384, 128])
        self.assertEqual([c[:3] for c in calls],
                         [('fused', 'bf16', 'int8'), ('gemm2', True, True)] * 2)

    def test_short_tail_does_not_reuse_previous_chunk_codes(self):
        calls = self.run_core([16384, 127])
        self.assertEqual([c[:3] for c in calls],
                         [('fused', 'bf16', 'int8'), ('gemm2', True, True),
                          ('silu',), ('gemm2', False, False)])

    def test_other_quantization_and_activation_contracts_stay_original(self):
        cases = [dict(dtype='fp16'), dict(use_int8_w4a8=False),
                 dict(block_shape=[128, 128]), dict(curr_a2_scale=object()),
                 dict(activation='gelu'), dict(is_gated=False),
                 dict(triton_quant=False),
                 dict(N=2560),
                 dict(gemm1_alpha=1.0), dict(gemm1_limit=7.0)]
        for case in cases:
            with self.subTest(case=case):
                calls = self.run_core([256], **case)
                self.assertFalse(any(c[0] == 'fused' for c in calls))
                self.assertEqual(calls[-1][1:3], (False, False))
                self.assertIs(calls[-1][3], case.get('curr_a2_scale'))

    def test_hip_quantization_stays_original_at_size_boundary(self):
        import os
        for backend, size, expected in [('hip', 50000000, False),
                                        ('auto', 24999999, False),
                                        ('auto', 25000000, True),
                                        ('triton', 1, True), ('legacy', 1, True)]:
            with self.subTest(backend=backend, size=size):
                with patch.dict(os.environ, {'AITER_PTQ_I8_BACKEND': backend,
                                              'AITER_PTQ_I8_HIP_MAX_ELEMS': '25000000'}):
                    self.assertEqual(TARGET._use_triton_quant(
                        SimpleNamespace(numel=lambda: size)), expected)

    def test_private_entry_unwrap_keeps_registered_entry_unchanged(self):
        def leaf(value):
            return value + 1
        def wrap(child):
            def wrapper(value):
                return child(value) * 2
            wrapper.__module__ = 'aiter.jit.utils.torch_guard'
            return wrapper
        original = wrap(wrap(leaf))
        leaf.__name__ = 'moe_c_fused_experts'
        with patch.object(TARGET, '_clone', return_value=lambda value: value + 10):
            private = TARGET._private_entry(original, {})
        self.assertEqual(original(1), 8)
        self.assertEqual(private(1), 11)


if __name__ == '__main__':
    unittest.main()
