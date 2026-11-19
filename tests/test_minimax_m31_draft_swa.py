# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hcu_sglang.layers.attention.minimax_m31_draft_swa import _DraftSwaForward


class TestMiniMaxDraftSwa(unittest.TestCase):
    def test_unsupported_shape_uses_existing_verify_fallback(self):
        orig = Mock()
        fwd = _DraftSwaForward(orig, 128)
        q = torch.empty((7, 64, 128), dtype=torch.bfloat16)
        k = torch.empty((7, 4, 128), dtype=torch.bfloat16)
        cache = torch.empty((128, 4, 128), dtype=torch.bfloat16)
        indptr = torch.tensor([0, 7], dtype=torch.int32)
        args = (q, k, k, q, cache, cache, indptr, indptr, indptr, None, True, None, 7, 1.0, 1.0, 1.0)
        self.assertIs(fwd(*args, sliding_window_size=4095), orig.return_value)
        orig.assert_called_once()
        self.assertEqual(orig.call_args.kwargs["sliding_window_size"], 4095)
        self.assertIsNone(fwd.kernel)
        self.assertFalse(_DraftSwaForward(None, 128)(*args, sliding_window_size=4095))


    def test_bidirectional_draft_preserves_visibility_in_split_kv(self):
        fwd = _DraftSwaForward(Mock(), 128)
        fwd.kernel = Mock()
        q = SimpleNamespace(shape=(7, 64, 128), dtype=torch.bfloat16,
                            is_cuda=True, stride=lambda dim: 1)
        k = torch.empty((7, 4, 128), dtype=torch.bfloat16)
        cache = torch.empty((128, 4, 128), dtype=torch.float8_e4m3fn)
        indptr = torch.tensor([0, 7], dtype=torch.int32)
        args = (q, k, k, q, cache, cache, indptr, indptr, indptr, None, False, None, 7, 1.0, 1.0, 1.0)
        self.assertTrue(fwd(*args, sliding_window_size=4095))
        fwd.fallback.assert_not_called()
        fwd.kernel.grow_buffers.assert_called_once_with(1)
        self.assertEqual(fwd.kernel.call_args.kwargs["sliding_window_size"], 4095)
        self.assertEqual(fwd.kernel.call_args.kwargs["page_size"], 128)
        self.assertFalse(fwd.kernel.call_args.kwargs["is_causal"])

        fwd.fallback.reset_mock()
        custom_mask = torch.ones((49,), dtype=torch.bool)
        masked_args = (*args[:9], custom_mask, *args[10:])
        self.assertIs(fwd(*masked_args, sliding_window_size=4095), fwd.fallback.return_value)
        fwd.fallback.assert_called_once()
        self.assertIs(fwd.fallback.call_args.args[9], custom_mask)
