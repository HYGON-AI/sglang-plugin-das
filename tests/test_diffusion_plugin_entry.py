import importlib.util
import os
import re
import sys
import types
import unittest
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import patch
from hcu_sglang.compat import aiter
from hcu_sglang.multimodal_gen import attention, minimax_h3, minimax_h3_quant_loader, minimax_h3_modulation

ROOT = Path(__file__).resolve().parents[1]
GROUP = "sglang.multimodal_gen.plugins"
PLUGIN = "sglang.multimodal_gen.runtime.platforms.plugins"
SRT = "sglang.srt.plugins.hook_registry"


class DiffusionEntryTests(unittest.TestCase):
    def setUp(self):
        official = Path(os.environ.get("OFFICIAL", str(ROOT.parent / "sglang-official-0.5.22")))
        self.base = official / "python"
        if not (self.base / "sglang/multimodal_gen/runtime/platforms/plugins.py").is_file():
            self.skipTest("fixed official Diffusion source required")
        self.modules = {}
        for name in (SRT, PLUGIN, "sglang.srt.environ"):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                key = ".".join(parts[:i])
                if key not in self.modules:
                    module = types.ModuleType(key)
                    module.__path__ = []
                    self.modules[key] = module
                if i > 1:
                    setattr(self.modules[".".join(parts[:i-1])], parts[i-1], self.modules[key])
        platform = self.modules[PLUGIN.rsplit(".", 1)[0]]
        platform.PLATFORM_PLUGINS_GROUP = "sglang.multimodal_gen.platforms"
        platform.get_selected_platform_dist = lambda: None
        self.allowed = "hcu"
        self.modules["sglang.srt.environ"].envs = types.SimpleNamespace(
            SGLANG_PLUGINS=types.SimpleNamespace(get=lambda: self.allowed))
        self.stack = patch.dict(sys.modules, self.modules)
        self.stack.start()
        self.addCleanup(self.stack.stop)
        self.srt = self.load(SRT)
        self.loader = self.load(PLUGIN)
        text = (ROOT / "pyproject.toml").read_text()
        entries = dict(re.findall(r'\[project.entry-points."([^"]+)"\]\s*\nhcu\s*=\s*"([^"]+)"', text))
        self.assertEqual(entries["sglang.srt.plugins"], "hcu_sglang:register")
        self.ep = EntryPoint(name="hcu", value=entries[GROUP], group=GROUP)
        self.discovery = patch.object(self.loader, "entry_points", side_effect=lambda *, group: (self.ep,) if group == GROUP else ())
        self.discovery.start()
        self.addCleanup(self.discovery.stop)

    def load(self, name):
        spec = importlib.util.spec_from_file_location(name, self.base / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
        spec.loader.exec_module(module)
        return module

    def test_discovery_registers_only_diffusion_hooks_once_without_target_imports(self):
        self.loader.load_plugins()
        hooks = self.loader.HookRegistry._hooks
        self.assertEqual(len(hooks), 18)
        self.assertFalse(self.srt.HookRegistry._hooks)
        self.assertTrue(all(t.startswith("sglang.multimodal_gen.") for t in hooks))
        self.assertIn("sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_gate", hooks)
        self.assertNotIn("sglang.multimodal_gen.runtime.models.dits.minimax_h3", sys.modules)
        for registered in hooks.values():
            self.assertEqual(len(registered), 1)
            self.assertEqual(registered[0][2].plugin_name, "hcu")
        self.loader.load_plugins()
        self.assertTrue(all(len(v) == 1 for v in hooks.values()))

    def test_whitelist_can_exclude_hcu(self):
        self.allowed = "other"
        self.loader.load_plugins()
        self.assertFalse(self.loader.HookRegistry._hooks)
        self.assertFalse(self.srt.HookRegistry._hooks)

    def test_actual_diffusion_apply_patches_consumer_and_leaves_srt_empty(self):
        from hcu_sglang.multimodal_gen import minimax_h3
        self.loader.load_plugins()
        with patch.dict(sys.modules):
            for target in self.loader.HookRegistry._hooks:
                parts = target.split(".")
                for i in range(1, len(parts)):
                    name = ".".join(parts[:i])
                    if name not in sys.modules:
                        module = types.ModuleType(name)
                        module.__path__ = []
                        sys.modules[name] = module
                        setattr(sys.modules[".".join(parts[:i-1])], parts[i-1], module)
                setattr(sys.modules[".".join(parts[:-1])], parts[-1], lambda *a, **k: "original")
            self.loader.apply_plugin_hooks()
            self.assertEqual(set(self.loader.HookRegistry._hooks), self.loader.HookRegistry._patched)
            module = sys.modules["sglang.multimodal_gen.runtime.models.dits.minimax_h3"]
            with patch.object(minimax_h3, "is_hcu", return_value=True):
                self.assertEqual(module._apply_qk_norm(2, 3, lambda x: x + 10, lambda x: x + 20, 128), (12, 23))
            self.assertFalse(self.srt.HookRegistry._patched)
