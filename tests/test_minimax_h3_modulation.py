import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from hcu_sglang.multimodal_gen import minimax_h3_modulation as mod

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
QUANT = "hcu_sglang.multimodal_gen.minimax_h3_quant"

class QuantConfig:
    compressed_adaln = True
    fused = True

class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.qkv_proj = None
    def forward(self, x, **kwargs):
        return x * 0.5

class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fc1 = None
    def forward(self, x):
        return x * 0.25

class Projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x * 0.75, None

class H3ModulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        official = Path(os.environ.get("OFFICIAL", str(ROOT.parent / "sglang-official-0.5.22")))
        cls.source = official / "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
        if not cls.source.is_file():
            raise unittest.SkipTest("fixed official MiniMax-H3 source required")
        spec = importlib.util.spec_from_file_location("h3_mod_registry", official / "python/sglang/srt/plugins/hook_registry.py")
        cls.registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.registry)

    def setUp(self):
        self.quant = types.ModuleType(QUANT)
        self.quant.MiniMaxH3QuantConfig = QuantConfig
        self.calls = []
        self.quant.supports_fp32_modulation = lambda *args, **kwargs: True
        def affine(x, shift, scale, indices):
            self.calls.append(("affine", x.dtype, shift.dtype))
            return (x.float() * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)
        def gate(x, g, other, indices):
            self.calls.append(("gate", x.dtype, g.dtype))
            return (x.float() + g.index_select(0, indices) * other.float()).to(x.dtype)
        self.quant.affine, self.quant.gate = affine, gate
        modules = {QUANT: self.quant, "sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant": self.quant}
        for name in (TARGET, "sglang.srt.plugins.hook_registry"):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                key = ".".join(parts[:i])
                modules.setdefault(key, types.ModuleType(key))
                modules[key].__path__ = []
                if i > 1:
                    setattr(modules[".".join(parts[:i-1])], parts[i-1], modules[key])
        modules["sglang.srt.plugins.hook_registry"] = self.registry
        self.target = modules[TARGET]
        stack = patch.dict(sys.modules, modules)
        stack.start()
        self.addCleanup(stack.stop)
        hcu = patch.object(mod, "is_hcu", return_value=True)
        hcu.start()
        self.addCleanup(hcu.stop)
        self.registry.HookRegistry.reset()
        self.addCleanup(self.registry.HookRegistry.reset)
        self.arch = types.SimpleNamespace(hidden_size=8, norm_eps=1e-6, final_norm_eps=1e-6,
            adaln_curve_grid=1025, latents_dim=2, patch_size=(1, 1, 1), audio_latents_dim=2)
        self.config = QuantConfig()
        generator = torch.Generator(device="cpu").manual_seed(42)
        self.x = torch.randn(5, 8, generator=generator).to(torch.bfloat16)
        self.params = tuple(torch.randn(3, 8, generator=generator) * 0.1 for _ in range(6))
        self.indices = torch.tensor([2, 0, 1, 2, 0])
        self.load(self.target.__dict__, self.source.read_text())
        self.old = {}
        self.load(self.old, (ROOT / "tests/fixtures/minimax_h3_legacy_modulation.py.txt").read_text())
        mod.register_hooks()
        self.registry.HookRegistry.apply_hooks()
        self.assertEqual(len(self.registry.HookRegistry._patched), 7)

    def load(self, namespace, text):
        namespace.update(torch=torch, nn=nn, _BF16_DTYPE=torch.bfloat16, _FP32_DTYPE=torch.float32,
            MiniMaxH3Attention=Attention, MiniMaxH3MLP=MLP, ColumnParallelLinear=Projection,
            MINIMAX_H3_ADALN_MODALITY_NUM=3, mark_minimax_h3_norm_modulate_site=lambda layer: None,
            minimax_h3_norm_modulate_active=lambda layer: True, _accepts_mxfp8_input=lambda layer: False)
        names = {"_norm", "_use_quant_fp32_modulation", "_modulate_scale_shift", "_modulate_gate",
            "_modulate_rmsnorm_scale_shift", "MiniMaxH3DiTBlock", "MiniMaxH3FinalLayer"}
        nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), "h3_cpu_consumer", "exec"), namespace)

    def test_real_block_forward_matches_legacy_and_preserves_residual(self):
        new = self.target.MiniMaxH3DiTBlock(self.arch, self.config, prefix="blocks.0", use_adaln_cache=True)
        old = self.old["MiniMaxH3DiTBlock"](self.arch, self.config, prefix="blocks.0", use_adaln_cache=True)
        kwargs = dict(adaln_input=None, combined_indices=self.indices, rope_cache=None,
            cu_seqlens=torch.tensor([0, 5], dtype=torch.int32), max_seqlen=5, adaln_params=self.params)
        before = self.x.clone()
        expected = old(self.x.clone(), **kwargs)
        self.calls.clear()
        actual = new(self.x, **kwargs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(self.x, before))
        self.assertEqual([x[0] for x in self.calls], ["affine", "gate", "affine", "gate"])
        self.assertTrue(all(x[1:] == (torch.bfloat16, torch.float32) for x in self.calls))
        self.assertFalse(mod._ACTIVE.get())

    def test_full_adaln_and_disabled_fusion_keep_legacy_norm_order(self):
        for compressed, fused in ((False, True), (True, False)):
            self.config.compressed_adaln, self.config.fused = compressed, fused
            self.arch.adaln_curve_grid = 1025 if compressed else None
            new = self.target.MiniMaxH3DiTBlock(self.arch, self.config, prefix="blocks.0", use_adaln_cache=True)
            old = self.old["MiniMaxH3DiTBlock"](self.arch, self.config, prefix="blocks.0", use_adaln_cache=True)
            kwargs = dict(adaln_input=None, combined_indices=self.indices, rope_cache=None,
                cu_seqlens=torch.tensor([0, 5], dtype=torch.int32), max_seqlen=5, adaln_params=self.params)
            expected = old(self.x.clone(), **kwargs)
            self.calls.clear()
            actual = new(self.x.clone(), **kwargs)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(new._hcu_legacy_quant_norm)
            self.assertFalse(new._hcu_fused_fp32_modulation)
            self.assertEqual(self.calls, [])
            self.assertFalse(mod._LEGACY_NORM.get())

    def test_real_final_layer_matches_legacy_and_keeps_fp32_outputs(self):
        new = self.target.MiniMaxH3FinalLayer(self.arch, self.config, prefix="final_layer", use_adaln_cache=True)
        old = self.old["MiniMaxH3FinalLayer"](self.arch, self.config, prefix="final_layer", use_adaln_cache=True)
        kwargs = dict(adaln_input=None, inverse_indices=self.indices, adaln_params=self.params[:2])
        expected = old(self.x.clone(), **kwargs)
        self.calls.clear()
        actual = new(self.x.clone(), **kwargs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(all(x.dtype == torch.float32 for x in actual))
        self.assertEqual([x[0] for x in self.calls], ["affine"])

    def test_constructor_requires_hcu_compressed_quant_and_opt_in(self):
        for hcu, config, curve in ((False, self.config, 1025), (True, None, 1025), (True, self.config, None)):
            mod.is_hcu.return_value = hcu
            self.arch.adaln_curve_grid = curve
            layer = types.SimpleNamespace()
            mod.after_layer_init(None, layer, self.arch, config)
            self.assertFalse(layer._hcu_fused_fp32_modulation)
        mod.is_hcu.return_value = True
        self.arch.adaln_curve_grid = 1025
        self.config.fused = False
        layer = types.SimpleNamespace()
        mod.after_layer_init(None, layer, self.arch, self.config)
        self.assertFalse(layer._hcu_fused_fp32_modulation)

    def test_nested_scope_and_exception_restore_context(self):
        enabled = types.SimpleNamespace(_hcu_fused_fp32_modulation=True)
        disabled = types.SimpleNamespace(_hcu_fused_fp32_modulation=False)
        def inner(layer):
            self.assertFalse(mod._ACTIVE.get())
            raise RuntimeError("probe")
        def outer(layer):
            self.assertTrue(mod._ACTIVE.get())
            with self.assertRaisesRegex(RuntimeError, "probe"):
                mod.around_layer_forward(inner, disabled)
            self.assertTrue(mod._ACTIVE.get())
        mod.around_layer_forward(outer, enabled)
        self.assertFalse(mod._ACTIVE.get())

    def test_unsupported_shape_preserves_original_arguments(self):
        self.quant.supports_fp32_modulation = lambda *args, **kwargs: False
        token = mod._ACTIVE.set(True)
        self.addCleanup(mod._ACTIVE.reset, token)
        seen = []
        result = mod.around_gate(lambda *args, **kwargs: seen.append((args, kwargs)) or "fallback",
            self.x, self.params[0], self.x, self.indices, dtype=torch.bfloat16, allow_inplace=False)
        self.assertEqual(result, "fallback")
        self.assertFalse(seen[0][1]["allow_inplace"])
        self.assertEqual(self.calls, [])

    def test_legacy_fixture_is_pinned(self):
        data = (ROOT / "tests/fixtures/minimax_h3_legacy_modulation.py.txt").read_bytes()
        manifest = json.loads((ROOT / "tests/contracts/minimax_h3_modulation_sources.json").read_text())
        self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["fixture_sha256"])

if __name__ == "__main__":
    unittest.main()
