import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


class DeepGemmPrefillAdmission(unittest.TestCase):
    def test_admission(self):
        source = pathlib.Path(__file__).resolve().parents[2] / "hcu_sglang/server_args_hooks.py"
        tree = ast.parse(source.read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "around_check_mega_moe_compat")
        active = {"hcu": True, "runtime": "deep_gemm", "provider": True}
        environment = types.ModuleType("sglang.srt.environ")
        environment.envs = types.SimpleNamespace(SGLANG_HCU_MEGA_MOE_RUNTIME=types.SimpleNamespace(get=lambda: active["runtime"]))
        alias = types.ModuleType("deep_gemm")
        names = []
        def find(name):
            self.assertIn(name, {"deepgemm", "megamoe"})
            names.append(name)
            return object() if active["provider"] else None
        namespace = {"is_hcu": lambda: active["hcu"], "_resolving_view": lambda x: x}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        function = namespace[node.name]
        view = types.SimpleNamespace(moe_a2a_backend="megamoe", enable_two_batch_overlap=False, enable_single_batch_overlap=False, disaggregation_mode="prefill")
        marker = object()
        original = lambda _: marker
        with patch.dict(sys.modules, {"sglang.srt.environ": environment, "deep_gemm": alias}), patch.object(importlib.util, "find_spec", find):
            self.assertIsNone(alias.__spec__)
            self.assertIsNone(function(original, view))
            self.assertEqual(names, ["deepgemm"])
            view.disaggregation_mode = "null"
            view.enable_prefill_cp = True
            view.attn_cp_size = 8
            self.assertIsNone(function(original, view))
            view.attn_cp_size = 1
            with self.assertRaises(ValueError):
                function(original, view)
            view.disaggregation_mode = "prefill"
            for runtime in ["deep_gemm", "megamoe"]:
                active.update(hcu=True, runtime=runtime, provider=True)
                self.assertIsNone(function(original, view))
                for field in ["enable_two_batch_overlap", "enable_single_batch_overlap"]:
                    setattr(view, field, True)
                    with self.assertRaises(ValueError):
                        function(original, view)
                    setattr(view, field, False)
                view.disaggregation_mode = "decode"
                with self.assertRaises(ValueError):
                    function(original, view)
                view.disaggregation_mode = "prefill"
                active["provider"] = False
                with self.assertRaises(RuntimeError):
                    function(original, view)
                active.update(provider=True, hcu=False)
                self.assertIs(function(original, view), marker)
            active.update(hcu=True, runtime="auto")
            self.assertIs(function(original, view), marker)
            active["runtime"] = "deep_gemm"
            view.moe_a2a_backend = "deepep"
            self.assertIs(function(original, view), marker)


if __name__ == "__main__":
    unittest.main()
