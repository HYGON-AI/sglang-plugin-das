import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.layers.quantization import channelwise_fp8 as runtime


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "HCU required")
class TestProjection(unittest.TestCase):
    def setUp(self):
        register_envs()
        torch.manual_seed(616)

    def reference(self, x, weight, scale):
        rows = x.float()
        scales = rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448
        q = (rows / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
        return (q.float() * scales) @ (weight.float() * scale).transpose(-1, -2)

    def tensors(self, rows, cols, groups=()):
        full = torch.randn((*groups, rows, cols), device="cuda")
        scales = full.abs().amax(-1, keepdim=True) / 448
        return (full / scales).clamp(-448, 448).to(torch.float8_e4m3fn), scales

    def test_fp32_lm_head(self):
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        w, s = self.tensors(256, 256)
        y = runtime.linear(x, w, s, output_dtype=torch.float32)
        ref = self.reference(x, w, s)
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)

    def test_tiled_lm_head(self):
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        w, s = self.tensors(32768, 256)
        ref = self.reference(x, w, s)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                y = runtime.linear(x, w, s, output_dtype=dtype)
                self.assertEqual(y.dtype, dtype)
                if dtype == torch.float32:
                    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
                else:
                    torch.testing.assert_close(y.float(), ref, rtol=0.005, atol=0.01)

    def test_markov_rank_slice(self):
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        w, s = self.tensors(512, 256)
        y = runtime.linear(x, w[256:], s[256:], output_dtype=torch.float32)
        ref = self.reference(x, w, s)[:, 256:]
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)

    def test_grouped_quant_order_and_graph_replay(self):
        w, s = self.tensors(128, 256, (8,))
        layer = SimpleNamespace(weight=w.flatten(0, 1), weight_scale_inv=s.flatten(0, 1))
        x = torch.randn(128, 8, 256, device="cuda", dtype=torch.bfloat16)
        fast = runtime.envs.SGLANG_DSV41_WO_A_QUANT_BEFORE_TRANSPOSE
        with patch.object(fast, "get", return_value=True):
            runtime.grouped(x, layer, 8, 128)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = runtime.grouped(x, layer, 8, 128)
        for magnitude in (1.0, 0.01):
            x.copy_(torch.randn_like(x) * magnitude)
            graph.replay()
            with patch.object(fast, "get", return_value=False):
                reference = runtime.grouped(x, layer, 8, 128)
            self.assertTrue(torch.equal(captured, reference))
            self.assertEqual(captured.stride(), reference.stride())

    def test_group_swap_fallback_for_misaligned_offset(self):
        x = torch.arange(49, device="cuda", dtype=torch.int16)[1:].view(2, 3, 8)
        self.assertTrue(torch.equal(runtime._swap_group_token(x), x.transpose(0, 1)))

    def test_grouped_wo_a(self):
        x = torch.randn(5, 2, 256, device="cuda", dtype=torch.bfloat16)
        w, s = self.tensors(128, 256, (2,))
        layer = SimpleNamespace(weight=w.flatten(0, 1), weight_scale_inv=s.flatten(0, 1))
        y = runtime.grouped(x, layer, 2, 128)
        ref = self.reference(x.transpose(0, 1), w, s).transpose(0, 1)
        torch.testing.assert_close(y.float(), ref, rtol=0.005, atol=0.01)


if __name__ == "__main__":
    unittest.main()
