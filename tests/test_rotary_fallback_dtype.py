# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import unittest

from hcu_sglang.layers.rotary import HcuRotaryFallback


class RotaryFallbackDtypeTests(unittest.TestCase):
    """The plugin wrapper converts the rotary cache to the query dtype; upstream keeps it fp32 on HIP."""

    def setUp(self):
        self.captured = []

    def original(self, positions, query, key, head_size, cos_sin_cache, is_neox):
        self.captured.append({"cos_sin_dtype": cos_sin_cache.dtype, "query_dtype": query.dtype})

    def make(self):
        return HcuRotaryFallback(self.original)

    def test_fp32_cache_is_converted_to_query_dtype(self):
        wrapper = self.make()
        positions = torch.arange(4)
        query = torch.zeros(4, 8, dtype=torch.bfloat16)
        cache = torch.zeros(64, 8, dtype=torch.float32)
        wrapper(positions, query, None, 8, cache, True)
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(self.captured[0]["cos_sin_dtype"], torch.bfloat16)
        self.assertEqual(self.captured[0]["query_dtype"], torch.bfloat16)
        self.assertEqual(cache.dtype, torch.float32)

    def test_matching_dtype_is_passed_through_untouched(self):
        wrapper = self.make()
        positions = torch.arange(4)
        query = torch.zeros(4, 8, dtype=torch.bfloat16)
        cache = torch.zeros(64, 8, dtype=torch.bfloat16)
        wrapper(positions, query, None, 8, cache, True)
        self.assertEqual(self.captured[0]["cos_sin_dtype"], torch.bfloat16)

    def test_converted_cache_is_memoised_across_calls(self):
        wrapper = self.make()
        positions = torch.arange(4)
        query = torch.zeros(4, 8, dtype=torch.bfloat16)
        cache = torch.zeros(64, 8, dtype=torch.float32)
        wrapper(positions, query, None, 8, cache, True)
        memo = wrapper._cache_memo
        wrapper(positions, query, None, 8, cache, True)
        self.assertIs(wrapper._cache_memo, memo)
        self.assertIs(wrapper._cache_memo[0], cache)


if __name__ == "__main__":
    unittest.main()
