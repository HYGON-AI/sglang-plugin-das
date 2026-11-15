from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import batch_overlap as P
from hcu_sglang.layers.moe import deepep_current as D
from sglang.srt.batch_overlap import single_batch_overlap as S
from sglang.srt.models import deepseek_v2 as M
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.test_utils import CustomTestCase

ORIGINAL_COMPUTE = (S.compute_overlap_args,)


class CombineOverlapTests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        P.register_hooks()
        HookRegistry.apply_hooks()

    def _combine(self, overlap, meta):
        result, event, hook = object(), object(), object()
        combine = Mock(return_value=(result, event, hook))
        impl = object.__new__(D._DeepEPDispatcherImplLowLatency)
        impl.overlap_args = overlap
        impl.meta_overlap_args = meta
        impl.packed_recv_count = object()
        impl.handle = object()
        impl.return_recv_hook = True
        impl._combine_input_ready_event_pending = False
        impl.device_module = SimpleNamespace(current_stream=lambda: Mock())
        impl._get_buffer = lambda: SimpleNamespace(low_latency_combine=combine)
        with patch.object(D, '_deepep_precompile_tp_barrier'), \
             patch.object(D, 'is_blackwell', return_value=False), \
             patch.object(D, 'use_groupgemm', True), \
             patch.object(torch.cuda, 'stream', return_value=nullcontext()):
            output = impl._combine_core(object(), object(), object())
        self.assertEqual(output, (result, event, hook))
        self.assertIsNone(impl.handle)
        return combine.call_args.kwargs

    def test_original_signal_contract_fails_before_native_combine(self):
        dispatch = SimpleNamespace(hidden_states=torch.empty(2, 64, 16, dtype=torch.int8))
        with patch.object(S.SboFlags, 'enable_combine_down_gemm_two_stream_overlap', return_value=True), \
             patch.object(S, 'is_blackwell', return_value=False), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(multi_processor_count=128)), \
             patch.object(torch.cuda, 'Event', return_value=Mock()), \
             patch.object(torch, 'zeros', return_value=object()):
            overlap, down, meta = ORIGINAL_COMPUTE[0](dispatch, Mock())
        self.assertIsNotNone(down)
        self.assertTrue(overlap.overlap)
        self.assertNotIn('block_m', meta)
        with self.assertRaisesRegex(KeyError, 'block_m'):
            self._combine(overlap, meta)

    def test_public_int8_compute_to_complete_combine(self):
        dispatch = SimpleNamespace(hidden_states=torch.empty(2, 64, 16, dtype=torch.int8))
        with patch.object(P, 'is_hcu', return_value=True), \
             patch.object(torch.cuda, 'get_device_properties', side_effect=AssertionError('device API')), \
             patch.object(torch.cuda, 'Event', side_effect=AssertionError('device event')):
            overlap, down, meta = M.compute_overlap_args(dispatch, Mock())
        self.assertEqual((overlap, down, meta), (None, None, {}))
        kwargs = self._combine(overlap, meta)
        self.assertTrue(kwargs['return_recv_hook'])
        self.assertFalse(kwargs['async_finish'])
        for key in ('overlap', 'comp_signal', 'packed_recv_count', 'block_m', 'threshold', 'num_sms'):
            self.assertNotIn(key, kwargs)

    def test_hcu_int8_and_fp8_skip_unsupported_signal_layouts(self):
        for dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            for shape in ((2, 64, 16), (0, 16)):
                original = Mock(side_effect=AssertionError('unsupported original'))
                dispatch = SimpleNamespace(hidden_states=torch.empty(shape, dtype=dtype))
                with self.subTest(dtype=dtype, shape=shape), patch.object(P, 'is_hcu', return_value=True):
                    self.assertEqual(P.around_compute_overlap_args(original, dispatch, None), (None, None, {}))
                original.assert_not_called()

    def test_hcu_normal_layout_skips_masked_only_overlap(self):
        original = Mock(side_effect=AssertionError('normal layout'))
        with patch.object(P, 'is_hcu', return_value=True):
            self.assertEqual(P.around_compute_overlap_args(original, SimpleNamespace(hidden_states=torch.empty(2, 16)), None), (None, None, {}))
        original.assert_not_called()

    def test_supported_hcu_layout_preserves_original_return_and_exception(self):
        dispatch = SimpleNamespace(hidden_states=torch.empty(2, 64, 16, dtype=torch.bfloat16))
        stream, value = object(), object()
        original = Mock(return_value=value)
        error = RuntimeError('original failure')
        with patch.object(P, 'is_hcu', return_value=True):
            self.assertIs(P.around_compute_overlap_args(original, dispatch, stream), value)
            original.assert_called_once_with(dispatch, stream)
            with self.assertRaises(RuntimeError) as caught:
                P.around_compute_overlap_args(Mock(side_effect=error), dispatch, stream)
        self.assertIs(caught.exception, error)

    def test_non_hcu_preserves_delegation_and_preimported_consumer(self):
        self.assertIs(M.compute_overlap_args, S.compute_overlap_args)
        dispatch, stream, value = object(), object(), object()
        original = Mock(return_value=value)
        with patch.object(P, 'is_hcu', return_value=False):
            self.assertIs(P.around_compute_overlap_args(original, dispatch, stream), value)
        original.assert_called_once_with(dispatch, stream)


if __name__ == '__main__':
    unittest.main()
