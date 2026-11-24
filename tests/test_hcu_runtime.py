# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class HcuRuntimeTests(unittest.TestCase):
    def test_int8_deepgemm_respects_explicit_disable_env(self):
        import importlib.util
        from types import ModuleType, SimpleNamespace

        def package(name):
            module = ModuleType(name)
            module.__path__ = []
            return module

        detection = ModuleType("hcu_sglang.utils.detection")
        detection.is_hcu = Mock(return_value=True)
        sglang_utils = ModuleType("sglang.srt.utils")
        sglang_utils.get_bool_env_var = lambda name: os.environ.get(
            name, "false"
        ).lower() in ("true", "1")
        moe = ModuleType("sglang.srt.layers.moe")
        moe.get_moe_runner_backend = Mock(
            return_value=SimpleNamespace(is_deep_gemm=lambda: True)
        )
        moe.get_moe_a2a_backend = Mock(
            return_value=SimpleNamespace(is_deepep=lambda: True)
        )
        modules = {
            "hcu_sglang.utils.detection": detection,
            "sglang": package("sglang"),
            "sglang.srt": package("sglang.srt"),
            "sglang.srt.utils": sglang_utils,
            "sglang.srt.layers": package("sglang.srt.layers"),
            "sglang.srt.layers.moe": moe,
        }
        spec = importlib.util.spec_from_file_location(
            "deepgemm_int8_under_test", ROOT / "hcu_sglang/layers/moe/deepgemm_int8.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            for disabled in ("0", "false"):
                with self.subTest(value=disabled), patch.dict(
                    os.environ, {"SGLANG_USE_DEEPGEMM_MOE": disabled}, clear=True
                ):
                    self.assertFalse(module.requested())
                    moe.get_moe_runner_backend.assert_not_called()
                    moe.get_moe_a2a_backend.assert_not_called()

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(module.requested())
            moe.get_moe_runner_backend.assert_called_once()
            moe.get_moe_a2a_backend.assert_called_once()


    def test_cold_import_keeps_startup_lightweight_and_uses_current_namespace(self):
        code = "import hcu_sglang,sys,json; print(json.dumps({'file':hcu_sglang.__file__,'loaded':[n for n in sys.modules if n.split('.')[0] in ['torch','aiter','sglang_hcu_plugin']]}))"
        report = json.loads(subprocess.check_output(
            [sys.executable, '-S', '-c', code], cwd=ROOT, text=True,
            env={**os.environ, 'PYTHONPATH': str(ROOT)},
        ))
        self.assertTrue(Path(report['file']).is_relative_to(ROOT / 'hcu_sglang'))
        self.assertEqual(report['loaded'], [])

    def test_launcher_preserves_spawn_bootstrap_entry_and_cli_arguments(self):
        from hcu_sglang.__main__ import main
        args = ['hcu-sglang-serve', '--model-path', 'test-model', '--tp-size', '8']
        calls = []
        with patch('runpy.run_module', side_effect=lambda *a, **kw: calls.append((a, kw, sys.argv[1:]))), \
             patch.object(sys, 'argv', args):
            main()
        self.assertEqual(calls, [(('hcu_sglang.launch_server',),
                                  {'run_name': '__main__', 'alter_sys': True}, args[1:])])

    def test_spawn_worker_imports_optional_aiter_api_after_compat_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for package in ['aiter', 'aiter/ops', 'aiter/ops/triton', 'sglang']:
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / '__init__.py').write_text('')
            (root / 'worker_fixture.py').write_text(
                "from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import fused_gemm_afp4wfp4_split_cat\n"
                "def worker(queue):\n"
                "    queue.put(fused_gemm_afp4wfp4_split_cat.__name__)\n")
            (root / 'sglang/launch_server.py').write_text(
                "import multiprocessing as mp\n"
                "from worker_fixture import worker\n"
                "if __name__ == '__main__':\n"
                "    ctx = mp.get_context('spawn')\n"
                "    queue = ctx.Queue()\n"
                "    process = ctx.Process(target=worker, args=(queue,))\n"
                "    process.start()\n"
                "    process.join(15)\n"
                "    assert process.exitcode == 0, process.exitcode\n"
                "    assert queue.get(timeout=5) == '_unsupported_fused_gemm'\n"
                "    print('spawn bootstrap passed')\n")
            env = {**os.environ, 'PYTHONPATH': str(root) + os.pathsep + str(ROOT)}
            # -S as in the sibling checks: the HCU image's site hooks cost tens of
            # seconds per interpreter, and the spawned worker starts with the same flag.
            result = subprocess.run([sys.executable, '-S', '-m', 'hcu_sglang'], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('spawn bootstrap passed', result.stdout)

    def test_cli_choices_include_all_plugin_quantization_methods_without_loading_configs(self):
        code = """
import argparse
import sys
import types
module = types.ModuleType('sglang.srt.arg_groups.arg_utils')
module.is_record = lambda cls: False
module.field_names = lambda cls: set(getattr(cls, '__annotations__', {}))
sys.modules[module.__name__] = module
from hcu_sglang.server_args import ensure_choices
from hcu_sglang.layers.quantization.registry import _HCU_QUANTIZATION_NAMES
class Args:
    ATTENTION_BACKEND_CHOICES = ['triton']
    DRAFT_ATTENTION_BACKEND_CHOICES = ['triton']
    QUANTIZATION_CHOICES = ['fp8']
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = []
    ServerArgs = type('ServerArgs', (), {})
    @classmethod
    def add_chunked_prefix_cache_attention_backend(cls, name):
        cls.CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(name)
assert ensure_choices(Args)
assert ensure_choices(Args)
parser = argparse.ArgumentParser()
parser.add_argument('--quantization', choices=Args.QUANTIZATION_CHOICES)
for name in _HCU_QUANTIZATION_NAMES:
    assert parser.parse_args(['--quantization', name]).quantization == name
    assert Args.QUANTIZATION_CHOICES.count(name) == 1
assert parser.parse_args(['--quantization', 'fp8']).quantization == 'fp8'
assert 'torch' not in sys.modules
assert not any(name.endswith('slimquant_w4a8_marlin') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                                env={**os.environ, 'PYTHONPATH': str(ROOT)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_does_not_import_legacy_namespace(self):
        for path in (ROOT / 'hcu_sglang').rglob('*.py'):
            tree = ast.parse(path.read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('sglang_hcu_plugin'), str(path))
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith('sglang_hcu_plugin') for alias in node.names), str(path))

    def test_scoped_launcher_replaces_global_usercustomize(self):
        self.assertFalse((ROOT / 'usercustomize.py').exists())
        self.assertFalse((ROOT / 'sitecustomize.py').exists())
        self.assertTrue((ROOT / 'hcu_sglang/__main__.py').is_file())
        self.assertTrue((ROOT / 'hcu_sglang/jit_overlay/csrc/deep_select/vendor/LICENSE').is_file())
        self.assertTrue(any((ROOT / 'hcu_sglang').rglob('*.hpp')))


if __name__ == '__main__':
    unittest.main()
