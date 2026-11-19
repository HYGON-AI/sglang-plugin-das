# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import itertools
import os
import unittest
from glm52_cpu_support import isolated_runtime

@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_EXPERT_MAP_GPU') == '1', 'GPU lane not requested')
class ExpertMapGpuTests(unittest.TestCase):
    def test_exact_mapping_and_fallback_layouts(self):
        with isolated_runtime() as rt:
            torch = rt.torch
            self.assertTrue(torch.cuda.is_available(), 'GPU lane requested without an available device')
            plugin = rt.load('hcu_sglang.layers.moe.standard_dispatcher', rt.root / 'hcu_sglang/layers/moe/standard_dispatcher.py')
            for ids_dtype, map_dtype, strided in itertools.product([torch.int32, torch.int64], [torch.int32, torch.int64], [False, True]):
                with self.subTest(ids_dtype=ids_dtype, map_dtype=map_dtype, strided=strided):
                    raw = torch.tensor([[-2, -1, 0, 1, 2, 4, 5, 999]], dtype=ids_dtype)
                    if strided:
                        ids = torch.empty(1, 16, dtype=ids_dtype, device='cuda')[:, ::2]
                        ids.copy_(raw)
                        mapping = torch.empty(10, dtype=map_dtype, device='cuda')[::2]
                    else:
                        ids = raw.cuda()
                        mapping = torch.empty(5, dtype=map_dtype, device='cuda')
                    mapping.copy_(torch.tensor([0, -1, 1, 2, 3], dtype=map_dtype))
                    out = plugin._map_topk_ids_to_local_experts(ids, mapping)
                    expected = torch.tensor([[-1, -1, 0, -1, 1, 3, -1, -1]], dtype=map_dtype)
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(ids.cpu(), raw, rtol=0, atol=0)
            empty = torch.empty(0, 8, dtype=torch.int32, device='cuda')
            self.assertEqual(tuple(plugin._map_topk_ids_to_local_experts(empty, mapping).shape), (0, 8))

    def test_graph_replay_reads_changed_ids_and_mapping(self):
        with isolated_runtime() as rt:
            torch = rt.torch
            self.assertTrue(torch.cuda.is_available(), 'GPU lane requested without an available device')
            plugin = rt.load('hcu_sglang.layers.moe.standard_dispatcher', rt.root / 'hcu_sglang/layers/moe/standard_dispatcher.py')
            for ids_dtype, map_dtype in itertools.product([torch.int32, torch.int64], [torch.int32, torch.int64]):
                ids = torch.tensor([[0, 1, -1, 4]], device='cuda', dtype=ids_dtype)
                mapping = torch.arange(5, device='cuda', dtype=map_dtype)
                plugin._map_topk_ids_to_local_experts(ids, mapping)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = plugin._map_topk_ids_to_local_experts(ids, mapping)
                for offset in [10, 30, 50]:
                    ids.copy_(torch.tensor([[2, -2, 99, 0]], dtype=ids_dtype))
                    mapping.copy_(torch.arange(5, dtype=map_dtype) + offset)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.tensor([[offset + 2, -1, -1, offset]], dtype=map_dtype)
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
