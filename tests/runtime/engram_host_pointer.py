import ctypes
import mmap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.test.test_utils import CustomTestCase

from hcu_sglang.layers import engram as P


class EngramPointerCPU(CustomTestCase):
    def tearDown(self):
        P._host_pointer_runtime.cache_clear()

    def test_fallback_preserves_return_and_exception(self):
        marker = object()
        error = RuntimeError("original exception")
        with patch.object(P, "is_hcu", return_value=False):
            self.assertIs(P.around_registered_device_ptr(lambda p: marker, 123), marker)
            with self.assertRaises(RuntimeError) as cm:
                P.around_registered_device_ptr(Mock(side_effect=error), 123)
            self.assertIs(cm.exception, error)

    def test_hcu_uses_mapped_address(self):
        def get_pointer(out, host, flags):
            self.assertEqual(host.value, 123)
            self.assertEqual(flags, 0)
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value = 456
            return 0

        with patch.object(P, "is_hcu", return_value=True), patch.object(
            P, "_host_pointer_runtime", return_value=SimpleNamespace(hipHostGetDevicePointer=get_pointer)
        ):
            original = Mock(side_effect=AssertionError("raw-address branch must not run"))
            self.assertEqual(P.around_registered_device_ptr(original, 123), 456)
            original.assert_not_called()

    def test_error_and_null_are_not_raw_address_fallbacks(self):
        for code in (0, 7):
            with self.subTest(code=code), patch.object(P, "is_hcu", return_value=True), patch.object(
                P, "_host_pointer_runtime", return_value=SimpleNamespace(hipHostGetDevicePointer=lambda *args: code)
            ):
                with self.assertRaisesRegex(RuntimeError, "hipHostGetDevicePointer failed"):
                    P.around_registered_device_ptr(Mock(), 123)

    def test_loaded_runtime_resolution_and_signature(self):
        fn = Mock()
        lib = SimpleNamespace(hipHostGetDevicePointer=fn)
        with patch.object(P, "find_loaded_library", side_effect=[None, "/loaded/libamdhip64.so"]) as find, patch.object(P.ctypes, "CDLL", return_value=lib):
            self.assertIs(P._host_pointer_runtime(), lib)
            self.assertEqual([x.args[0] for x in find.call_args_list], ["libgalaxyhip", "libamdhip64"])
            self.assertIs(fn.restype, ctypes.c_int)
            self.assertEqual(fn.argtypes, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint])

    def test_missing_runtime_is_rejected(self):
        with patch.object(P, "find_loaded_library", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "HIP runtime is not loaded"):
                P._host_pointer_runtime()


class EngramPointerGPU(CustomTestCase):
    def test_public_host_gather_and_graph_updates(self):
        from sglang.srt.plugins import load_plugins

        load_plugins()
        from sglang.srt.layers import engram as N
        from sglang.srt.plugins.hook_registry import HookRegistry

        self.assertTrue(P.is_hcu())
        self.assertIn("sglang.srt.layers.engram._registered_device_ptr", HookRegistry._patched)
        self.assertEqual(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx936")
        mm = mmap.mmap(-1, 4096)
        raw = torch.frombuffer(mm, dtype=torch.uint8)
        host = raw.data_ptr()
        rt = torch.cuda.cudart()
        registered = False
        try:
            self.assertEqual(int(rt.cudaHostRegister(host, 4096, 0)), 0)
            registered = True
            device = N._registered_device_ptr(host)
            self.assertGreater(device, 0)
            ids = torch.empty(6, dtype=torch.int64, device="cuda")
            out = torch.empty((6, 128), dtype=torch.bfloat16, device="cuda")
            graph = None
            for state in range(3):
                values = ((torch.arange(16 * 128).reshape(16, 128) + state) % 17 - 8).to(torch.float8_e4m3fn)
                raw[:2048].copy_(values.view(torch.uint8).reshape(-1))
                exponent = 127 + state
                raw[2048:2064].fill_(exponent)
                cpu_ids = torch.tensor([-1, state, 7, 15, 16, 1])
                ids.copy_(cpu_ids)
                reference = (values[cpu_ids.clamp(0, 15)].float() * (2.0 ** (exponent - 127))).to(torch.bfloat16)
                reference[(cpu_ids < 0) | (cpu_ids >= 16)] = 0
                N.engram_gather(device, device + 2048, ids, out, 128, 128, 0, 16)
                torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
                eager = out.clone()
                if graph is None:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        N.engram_gather(device, device + 2048, ids, out, 128, 128, 0, 16)
                graph.replay()
                torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
                torch.testing.assert_close(out, eager, rtol=0, atol=0)
            torch.cuda.synchronize()
            del graph
        finally:
            if registered:
                code = int(rt.cudaHostUnregister(host))
                self.assertEqual(code, 0)
                if code == 0:
                    del raw
                    mm.close()
            else:
                del raw
                mm.close()


if __name__ == "__main__":
    unittest.main()
