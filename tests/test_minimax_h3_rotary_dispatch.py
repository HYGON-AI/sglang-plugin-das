# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hcu_sglang.multimodal_gen import minimax_h3


class MiniMaxH3RotaryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.query = torch.zeros(3, 2, 8)
        self.key = torch.zeros_like(self.query)
        self.cache = torch.zeros(16, 8)
        self.positions = torch.arange(3)

    def test_native_keyword_preserves_identity_on_both_platforms(self):
        rotary = object()
        seen = []

        def native(query, key, cache, positions, *, rotary_emb):
            seen.append((query, key, cache, positions, rotary_emb))
            return query, key

        for hcu in (False, True):
            with self.subTest(hcu=hcu), patch.object(minimax_h3, 'is_hcu', return_value=hcu):
                result = minimax_h3.around_apply_rope_qk(
                    native, self.query, self.key, self.cache, self.positions,
                    rotary_emb=rotary,
                )
                self.assertIs(result[0], self.query)
                self.assertIs(result[1], self.key)
                self.assertEqual(len(seen), 1)
                for actual, expected in zip(seen.pop(), (
                    self.query, self.key, self.cache, self.positions, rotary,
                )):
                    self.assertIs(actual, expected)

    def test_explicit_none_is_forwarded(self):
        seen = []

        def native(query, key, cache, positions, *, rotary_emb):
            seen.append(rotary_emb)
            return query, key

        with patch.object(minimax_h3, 'is_hcu', return_value=True):
            minimax_h3.around_apply_rope_qk(
                native, self.query, self.key, self.cache, self.positions,
                rotary_emb=None,
            )
        self.assertEqual(seen, [None])

    def test_native_error_is_not_swallowed(self):
        def native(query, key, cache, positions, *, rotary_emb):
            raise RuntimeError('native rotary contract failed')

        with patch.object(minimax_h3, 'is_hcu', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'native rotary contract failed'):
                minimax_h3.around_apply_rope_qk(
                    native, self.query, self.key, self.cache, self.positions,
                    rotary_emb=object(),
                )

    def test_legacy_cpu_call_does_not_gain_a_keyword(self):
        def legacy(query, key, cache, positions):
            return query, key

        for hcu in (False, True):
            with self.subTest(hcu=hcu), patch.object(minimax_h3, 'is_hcu', return_value=hcu):
                result = minimax_h3.around_apply_rope_qk(
                    legacy, self.query, self.key, self.cache, self.positions,
                )
                self.assertIs(result[0], self.query)
                self.assertIs(result[1], self.key)


    def test_qk_norm_uses_upstream_probe_on_gfx936(self):
        query = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:2"))
        key = SimpleNamespace(dtype=torch.bfloat16)
        q_norm, k_norm = Mock(), Mock()
        original = Mock()
        target = "sglang.kernels.ops.layernorm.norm"
        probe = Mock()
        with (
            patch.object(minimax_h3, "is_hcu", return_value=True),
            patch.object(minimax_h3, "get_gcn_arch") as arch,
            patch.dict("sys.modules", {target: SimpleNamespace(can_use_fused_inplace_qknorm=probe)}),
        ):
            arch.return_value = "gfx936:sramecc+"
            probe.return_value = True
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertIs(result, original.return_value)
            probe.assert_called_once_with(128, torch.bfloat16)
            arch.assert_called_once_with(2)
            original.assert_called_once_with(query, key, q_norm, k_norm, 128)

            original.reset_mock()
            probe.return_value = False
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertEqual(result, (q_norm.return_value, k_norm.return_value))
            original.assert_not_called()

            probe.reset_mock()
            arch.return_value = "gfx938"
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertEqual(result, (q_norm.return_value, k_norm.return_value))
            probe.assert_not_called()
            original.assert_not_called()


if __name__ == '__main__':
    unittest.main()
