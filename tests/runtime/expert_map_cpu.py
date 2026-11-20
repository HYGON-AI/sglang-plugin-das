# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import itertools
import unittest
from glm52_cpu_support import isolated_runtime

class ExpertMapCpuTests(unittest.TestCase):
    def test_fallback_preserves_bounds_sentinel_and_source(self):
        with isolated_runtime() as rt:
            torch = rt.torch
            plugin = rt.load('hcu_sglang.layers.moe.standard_dispatcher', rt.root / 'hcu_sglang/layers/moe/standard_dispatcher.py')
            for ids_dtype, map_dtype in itertools.product([torch.int32, torch.int64], [torch.int32, torch.int64]):
                with self.subTest(ids_dtype=ids_dtype, map_dtype=map_dtype):
                    ids = torch.tensor([[-2, -1, 0, 1, 2, 3, 4, 999]], dtype=ids_dtype)
                    mapping = torch.tensor([0, -1, 1, 2, -1], dtype=map_dtype)
                    before = ids.clone()
                    result = plugin._map_topk_ids_to_local_experts(ids, mapping)
                    expected = torch.tensor([[-1, -1, 0, -1, 1, 2, -1, -1]], dtype=map_dtype)
                    self.assertEqual(result.dtype, map_dtype)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    torch.testing.assert_close(ids, before, rtol=0, atol=0)

    def test_strided_views_and_empty_ids(self):
        with isolated_runtime() as rt:
            torch = rt.torch
            plugin = rt.load('hcu_sglang.layers.moe.standard_dispatcher', rt.root / 'hcu_sglang/layers/moe/standard_dispatcher.py')
            ids = torch.tensor([[0, 99, 1, 99], [-1, 99, 2, 99]])[:, ::2]
            mapping = torch.tensor([4, 99, -1, 99, 8, 99])[::2]
            self.assertFalse(ids.is_contiguous())
            self.assertFalse(mapping.is_contiguous())
            result = plugin._map_topk_ids_to_local_experts(ids, mapping)
            torch.testing.assert_close(result, torch.tensor([[4, -1], [-1, 8]]), rtol=0, atol=0)
            for shape in [(0, 8), (2, 0)]:
                empty = torch.empty(shape, dtype=torch.int64)
                out = plugin._map_topk_ids_to_local_experts(empty, mapping)
                self.assertEqual(tuple(out.shape), shape)

    def test_cpu_does_not_initialize_cuda(self):
        with isolated_runtime() as rt:
            self.assertFalse(rt.torch.cuda.is_initialized())
            plugin = rt.load('hcu_sglang.layers.moe.standard_dispatcher', rt.root / 'hcu_sglang/layers/moe/standard_dispatcher.py')
            plugin._map_topk_ids_to_local_experts(rt.torch.tensor([0, -1]), rt.torch.tensor([3]))
            self.assertFalse(rt.torch.cuda.is_initialized())

if __name__ == '__main__':
    unittest.main()
