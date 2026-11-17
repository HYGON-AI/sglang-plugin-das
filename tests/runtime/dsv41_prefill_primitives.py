import importlib.util
import os
from pathlib import Path
import unittest

import torch
from sglang.test.test_utils import CustomTestCase


ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in")
class TestPrefillPrimitives(CustomTestCase):
    def test_fp4_math_graph_and_streams(self):
        native = load("fp4_native_test", "hcu_sglang/models/deepseek/current_v4/hcu_backend/fp4_rope_native.py")
        quant = load("fp4_reference_test", "hcu_sglang/models/deepseek/current_v4/hcu_backend/torch_quant.py")
        torch.manual_seed(20261116)

        def reference(x, f, compressed):
            tail = torch.view_as_complex(x[..., -64:].float().unflatten(-1, (-1, 2)).contiguous())
            rotated = tail * f.view(x.shape[0], *[1] * (x.ndim - 2), 32)
            y = torch.cat([x[..., :-64], torch.view_as_real(rotated).flatten(-2).to(x.dtype)], -1)
            return (quant.fake_quant_compressed_kv if compressed else quant.fake_quant_fp4)(y)

        for compressed in (False, True):
            for shape in ((0, 128), (1, 128), (31, 8, 128), (37, 512), (4096, 512)):
                x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                f = torch.polar(torch.ones(shape[0], 32, device="cuda"), torch.randn(shape[0], 32, device="cuda"))
                torch.testing.assert_close(native.rope_fp4_fake_quant(x, f, 64, compressed), reference(x, f, compressed), rtol=0, atol=0)
            x = torch.randn(8, 2, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
            x[0, 0, 0], x[1, 0, 0], x[2, 0, 0] = float("nan"), float("inf"), -float("inf")
            f = torch.ones(8, 32, device="cuda", dtype=torch.complex64)
            torch.testing.assert_close(native.rope_fp4_fake_quant(x, f, 64, compressed), reference(x, f, compressed), rtol=0, atol=0, equal_nan=True)
            x = torch.randn(16, 2, 128, device="cuda", dtype=torch.bfloat16)
            f = torch.ones(16, 32, device="cuda", dtype=torch.complex64)
            native.rope_fp4_fake_quant(x, f, 64, compressed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = native.rope_fp4_fake_quant(x, f, 64, compressed)
            for _ in range(3):
                x.copy_(torch.randn_like(x))
                f.copy_(torch.polar(torch.ones(16, 32, device="cuda"), torch.randn(16, 32, device="cuda")))
                captured.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(captured, reference(x, f, compressed), rtol=0, atol=0)
            a, b = torch.randn_like(x), torch.randn_like(x)
            first, second = torch.cuda.Stream(), torch.cuda.Stream()
            first.wait_stream(torch.cuda.current_stream())
            second.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(first):
                ya = native.rope_fp4_fake_quant(a, f, 64, compressed)
            with torch.cuda.stream(second):
                yb = native.rope_fp4_fake_quant(b, f, 64, compressed)
            torch.cuda.current_stream().wait_stream(first)
            torch.cuda.current_stream().wait_stream(second)
            torch.testing.assert_close(ya, reference(a, f, compressed), rtol=0, atol=0)
            torch.testing.assert_close(yb, reference(b, f, compressed), rtol=0, atol=0)

    def test_rope_flat_strided(self):
        from sglang.kernels.ops.attention import deepseek_v4_rope as rope

        old = rope._USE_BATCHED_ROPE
        try:
            for inverse in (False, True):
                x = torch.randn(31, 128, 192, device="cuda", dtype=torch.bfloat16)[..., -64:]
                expected = x.clone()
                f = torch.polar(torch.ones(40, 32, device="cuda"), torch.randn(40, 32, device="cuda"))
                p = torch.randperm(40, device="cuda")[:31]
                rope.set_batched_rope(False)
                rope.apply_rotary_emb_triton(expected, f, p, inverse=inverse)
                rope.set_batched_rope(True)
                self.assertIs(rope.apply_rotary_emb_triton(x, f, p, inverse=inverse), x)
                torch.testing.assert_close(x, expected, rtol=2**-7, atol=2**-20)
        finally:
            rope.set_batched_rope(old)


if __name__ == "__main__":
    unittest.main()
