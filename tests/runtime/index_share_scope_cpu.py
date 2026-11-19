# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import itertools
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]

def initializer():
    source = Path(os.environ['SGLANG_SOURCE_ROOT']) / 'sglang/srt'
    worker = ast.parse((source / 'speculative/eagle_worker_v2.py').read_text())
    model = ast.parse((source / 'configs/model_config.py').read_text())
    names = {'get_dsa_index_kpool', 'get_dsa_mtp_topk_width'}
    nodes = [x for x in model.body if isinstance(x, ast.FunctionDef) and x.name in names]
    cls = next(x for x in worker.body if isinstance(x, ast.ClassDef) and x.name == 'EagleDraftWorker')
    nodes.append(next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_init_dsa_index_share_state'))
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {}
    exec(compile(unit, str(source), 'exec'), namespace)
    return namespace['_init_dsa_index_share_state']

def hook_module():
    detection = ModuleType('hcu_sglang.utils.detection')
    detection.is_hcu = lambda: True
    previous = sys.modules.get(detection.__name__)
    sys.modules[detection.__name__] = detection
    try:
        path = ROOT / 'hcu_sglang/models/deepseek/deepseek_nextn/methods/index_share.py'
        spec = importlib.util.spec_from_file_location('index_share_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(detection.__name__, None)
        else:
            sys.modules[detection.__name__] = previous

class IndexShareScopeTests(unittest.TestCase):
    def test_hook_preserves_initializer_fields_and_return(self):
        init, plugin = initializer(), hook_module()
        for hcu, glm, topk, enabled, width, kpool in itertools.product([False, True], [False, True], [1, 2], [False, True], [None, 2048, 4096], [1, 4]):
            with self.subTest(hcu=hcu, glm=glm, topk=topk, enabled=enabled, width=width, kpool=kpool):
                cfg = SimpleNamespace(model_type='glm_moe_dsa' if glm else 'deepseek_v3', index_share_for_mtp_iteration=enabled, index_topk=width, index_kpool=kpool)
                worker = SimpleNamespace(topk=topk, draft_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=cfg)))
                init(worker)
                before = (worker.index_share_for_mtp_iteration, worker.dsa_index_topk, worker.dsa_seed_topk_width)
                old_seed = worker.seed_dsa_topk_from_draft_extend
                plugin.is_hcu = lambda: hcu
                result = object()
                self.assertIs(plugin.after_init_dsa_index_share_state(result, worker), result)
                self.assertEqual(before, (worker.index_share_for_mtp_iteration, worker.dsa_index_topk, worker.dsa_seed_topk_width))
                self.assertEqual(worker.seed_dsa_topk_from_draft_extend, False if hcu and glm else old_seed)

    def test_registration_target_and_callback(self):
        plugin = hook_module()
        calls = []
        names = ['sglang', 'sglang.srt', 'sglang.srt.plugins', 'sglang.srt.plugins.hook_registry']
        saved = {name: sys.modules.get(name) for name in names}
        try:
            for name in names:
                module = ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
                if '.' in name:
                    parent, child = name.rsplit('.', 1)
                    setattr(sys.modules[parent], child, module)
            registry = sys.modules[names[-1]]
            registry.HookRegistry = SimpleNamespace(register=lambda *args: calls.append(args))
            registry.HookType = SimpleNamespace(AFTER=object())
            self.assertEqual(plugin.register_hooks(), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 'sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._init_dsa_index_share_state')
            self.assertIs(calls[0][1], plugin.after_init_dsa_index_share_state)
            self.assertIs(calls[0][2], registry.HookType.AFTER)
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

    def test_missing_model_type_preserves_seed(self):
        plugin = hook_module()
        worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True, draft_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace())))
        token = object()
        self.assertIs(plugin.after_init_dsa_index_share_state(token, worker), token)
        self.assertTrue(worker.seed_dsa_topk_from_draft_extend)

if __name__ == '__main__':
    unittest.main()
