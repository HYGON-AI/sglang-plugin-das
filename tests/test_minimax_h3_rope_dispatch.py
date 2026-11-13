import sys
import types
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.multimodal_gen import minimax_h3 as h3


class VisibleGpuView:
    is_cuda = True
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape
    def view(self, *shape):
        return self.tensor.view(*shape)


class H3RopeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.q = VisibleGpuView(torch.ones(2, 3, 128))
        self.k = VisibleGpuView(torch.ones(2, 2, 128))
        self.positions = torch.arange(2)
        self.cache = torch.ones(2, 96)

    def test_adapter_surface_is_used_when_native_symbol_is_absent(self):
        seen = []
        adapter = types.ModuleType("sgl_kernel")
        adapter.minimax_h3_rotary_embedding = lambda *args: seen.append(args)
        with patch.object(h3, "is_hcu", return_value=True), patch.object(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", None, create=True), patch.dict(sys.modules, {"sgl_kernel": adapter}):
            result = h3.around_apply_rope_qk(lambda *args: self.fail("unexpected original"), self.q, self.k, self.cache, self.positions)
        self.assertEqual(result, (self.q, self.k))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1].shape, (2, 384))
        self.assertEqual(seen[0][2].shape, (2, 256))
        self.assertEqual(seen[0][3], 128)
        self.assertIs(seen[0][4], self.cache)
        self.assertTrue(seen[0][5])

    def test_native_provider_remains_preferred(self):
        seen = []
        adapter = types.ModuleType("sgl_kernel")
        adapter.minimax_h3_rotary_embedding = lambda *args: self.fail("native provider bypassed")
        with patch.object(h3, "is_hcu", return_value=True), patch.object(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", lambda *args: seen.append(args), create=True), patch.dict(sys.modules, {"sgl_kernel": adapter}):
            h3.around_apply_rope_qk(lambda *args: self.fail(), self.q, self.k, self.cache, self.positions)
        self.assertEqual(len(seen), 1)

    def test_non_hcu_preserves_original(self):
        with patch.object(h3, "is_hcu", return_value=False):
            self.assertEqual(h3.around_apply_rope_qk(lambda *args: "original", self.q, self.k, self.cache, self.positions), "original")

if __name__ == "__main__":
    unittest.main()
