import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from sglang.test.test_utils import CustomTestCase

ROOT = Path(os.environ.get("IFB_STAGE", str(Path(__file__).resolve().parents[2])))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load("hcu_sglang.layers.quantization.dsv41_int8_native", ROOT / "hcu_sglang/layers/quantization/dsv41_int8_native.py")
Q = load("native_projection_under_test", ROOT / "hcu_sglang/layers/quantization/channelwise_int8.py")
CASES = []


class TestGroupedNative(CustomTestCase):
    def test_cpu_and_unsupported_inputs(self):
        x = torch.randn(3, 2, 32)
        w = torch.randint(-127, 128, (64, 32), dtype=torch.int8)
        scales = torch.rand(64, 1)
        self.assertFalse(N.supported(x, 2, 32, x.dtype))
        with mock.patch.object(N, "library", side_effect=AssertionError("CPU imported native")):
            y = Q._grouped(x, w, scales, 2, 32)
        self.assertEqual(y.shape, (3, 2, 32))
        with self.assertRaises(ValueError):
            Q._grouped(x, w.float(), scales, 2, 32)
        self.assertEqual(Q._grouped(x[:0], w, scales, 2, 32).shape, (0, 2, 32))

    @unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in")
    def test_native_numbers_graph_and_streams(self):
        torch.manual_seed(20261116)
        self.assertEqual(N._architecture(0), "gfx936")
        w = torch.randint(-127, 128, (8192, 4096), device="cuda", dtype=torch.int8)
        scales = torch.rand(8192, 1, device="cuda") * .001 + .0001

        def reference(x):
            with mock.patch.object(N, "supported", return_value=False):
                return Q._grouped(x, w, scales, 8, 1024)

        def compare(x, name):
            expected = reference(x)
            with mock.patch.object(N, "project", wraps=N.project) as actual_call:
                actual = Q._grouped(x, w, scales, 8, 1024)
            self.assertEqual(actual_call.call_count, 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            CASES.append({"name": name, "passed": True, "shape": list(x.shape), "max_abs_error": float((actual.float() - expected.float()).abs().nan_to_num().max())})
            return actual

        for m in (1, 2, 15, 16, 31, 32, 4096):
            compare(torch.randn(m, 8, 4096, device="cuda", dtype=torch.bfloat16), "random_" + str(m))
        compare(torch.zeros(2, 8, 4096, device="cuda", dtype=torch.bfloat16), "zero")
        ties = torch.zeros(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
        ties[..., :5] = torch.tensor([127., .5, -.5, 1.5, -1.5], device="cuda", dtype=torch.bfloat16)
        compare(ties, "half_away")
        compare(torch.randn(2, 8, 8192, device="cuda", dtype=torch.bfloat16)[..., ::2], "input_stride")

        x = torch.randn(16, 8, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            Q._grouped(x, w, scales, 8, 1024)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = Q._grouped(x, w, scales, 8, 1024)
        for i in range(3):
            x.copy_(torch.randn_like(x))
            w.copy_(torch.randint_like(w, -127, 128))
            scales.copy_(torch.rand_like(scales) * .001 + .0001)
            captured.fill_(float("nan"))
            expected = reference(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            CASES.append({"name": "graph_update_" + str(i), "passed": True, "max_abs_error": 0.})

        a = torch.randn_like(x)
        b = torch.randn_like(x)
        expected_a, expected_b = reference(a), reference(b)
        stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
        stream_a.wait_stream(torch.cuda.current_stream())
        stream_b.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_a):
            actual_a = Q._grouped(a, w, scales, 8, 1024)
        with torch.cuda.stream(stream_b):
            actual_b = Q._grouped(b, w, scales, 8, 1024)
        torch.cuda.current_stream().wait_stream(stream_a)
        torch.cuda.current_stream().wait_stream(stream_b)
        torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
        CASES.append({"name": "two_streams", "passed": True})


if __name__ == "__main__":
    unittest.main()
