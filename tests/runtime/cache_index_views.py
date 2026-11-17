# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.mem_cache import cache_index_views as views


class FullIndexFallbackTests(unittest.TestCase):
    def test_cpu_and_non_hcu_preserve_original(self):
        value = torch.arange(8, dtype=torch.int64)
        ct = object()
        child = SimpleNamespace(component_data={ct: SimpleNamespace(value=value)})
        component = SimpleNamespace(component_type=ct)
        parent = SimpleNamespace(key=range(6))
        for hcu in (False, True):
            token, calls = object(), []

            def original(*args):
                calls.append(args)
                return token

            with patch.object(views, "is_hcu", return_value=hcu):
                self.assertIs(views.around_full_index_split(original, component, parent, child), token)
            self.assertEqual(calls, [(component, parent, child)])


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in required")
class FullIndexTreeGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import window_tree_cpu

        cls.fixture = window_tree_cpu
        cls.fixture.WindowTreeCpuTests.setUpClass()

    def test_real_tree_split_preserves_owned_indices_and_host_lock(self):
        from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
            CpCacheLayerSplitDeepSeekV4TokenToKVPool,
        )

        helper = self.fixture.WindowTreeCpuTests()
        ct = self.fixture.N.ComponentType.FULL
        for prefix, backing in ((6, 8), (4, 8), (2, 8), (6, 256)):
            tree, key, node = helper.tree(device="cuda")
            pool = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
            tree.components_by_type[ct].cache.token_to_kv_pool_allocator._kvcache = pool
            request = torch.arange(8, dtype=torch.int64, device="cuda")
            cd = node.component_data[ct]
            cd.value = request.clone() if backing == 8 else torch.arange(backing, device="cuda")[:8]
            cd.metadata["host_uuid"] = "test-host-lock"
            cd.host_value = torch.arange(8, dtype=torch.int64)
            owned = cd.value
            parent, _ = tree._split_node(key, node, prefix)
            request.fill_(-1)
            left, right = parent.component_data[ct], node.component_data[ct]
            torch.testing.assert_close(torch.cat([left.value, right.value]), torch.arange(8, device="cuda"), rtol=0, atol=0)
            self.assertEqual(left.value.data_ptr() == owned.data_ptr(), prefix >= 4 and backing <= 2 * prefix)
            self.assertNotEqual(right.value.data_ptr(), owned[prefix:].data_ptr())
            self.assertEqual(left.metadata["host_uuid"], "test-host-lock")
            self.assertNotIn("host_uuid", right.metadata)
            torch.testing.assert_close(torch.cat([left.host_value, right.host_value]), torch.arange(8), rtol=0, atol=0)
            self.assertEqual(left.lock_ref, right.lock_ref)
            self.assertEqual(left.session_ref, right.session_ref)
            spans = [n.component_data[ct].value for n, _, _ in tree._walk_span(key, 8)]
            torch.testing.assert_close(torch.cat(spans), torch.arange(8, device="cuda"), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
