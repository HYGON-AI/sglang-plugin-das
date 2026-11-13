import sys
import ast
import os
import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hcu_sglang.multimodal_gen import minimax_h3_quant_loader as loader

ROOT = Path(__file__).resolve().parents[1]


class H3QuantLoaderTests(unittest.TestCase):
    def setUp(self):
        self.config = object()
        self.quant = types.ModuleType("hcu_sglang.multimodal_gen.minimax_h3_quant")
        self.quant.inspect_int8_checkpoint = lambda paths: {"profile": "backbone-adaln"}
        self.quant.inspect_fp8_checkpoint = lambda paths: None
        self.quant.inspect_quant_checkpoint = lambda paths, variant: (self.config, False, (1025, 8))
        self.args = types.SimpleNamespace(
            transformer_weights_path="/weights", model_variant="fl2va",
            minimax_h3_adaln_cache_path=None, minimax_h3_adaln_online=False,
            should_use_fsdp_for_component=lambda component: False,
        )
        self.arch = types.SimpleNamespace(checkpoint_uses_diffusers_layout=True)
        self.kwargs = dict(
            model_cls=type("MiniMaxH3DiTModel", (), {}), server_args=self.args,
            safetensors_list=["weights.safetensors"], arch_config=self.arch,
            checkpoint_quant_config=None, component_name="transformer", extra=object(),
        )
        self.stack = patch.dict(sys.modules, {self.quant.__name__: self.quant})
        self.stack.start()
        self.addCleanup(self.stack.stop)
        self.hcu = patch.object(loader, "is_hcu", return_value=True)
        self.hcu.start()
        self.addCleanup(self.hcu.stop)

    def test_quant_config_and_curve_reach_original_without_losing_arguments(self):
        received = {}
        def original(**kwargs):
            received.update(kwargs)
            return "plan"
        self.assertEqual(loader.around_resolve_quant_load_spec(original, **self.kwargs), "plan")
        self.assertIs(received["checkpoint_quant_config"], self.config)
        self.assertIs(received["extra"], self.kwargs["extra"])
        self.assertFalse(self.arch.checkpoint_uses_diffusers_layout)
        self.assertEqual((self.arch.adaln_curve_grid, self.arch.time_embed_dim), (1025, 8))
        self.assertIsNone(self.kwargs["checkpoint_quant_config"])

    def test_unrecognized_checkpoint_preserves_adapter_original(self):
        self.quant.inspect_quant_checkpoint = lambda *args: None
        token = object()
        self.assertIs(loader.around_resolve_quant_load_spec(lambda **kwargs: token, **self.kwargs), token)
        self.assertTrue(self.arch.checkpoint_uses_diffusers_layout)

    def test_non_hcu_does_not_inspect_checkpoint(self):
        loader.is_hcu.return_value = False
        self.quant.inspect_quant_checkpoint = lambda *args: self.fail("unexpected inspection")
        self.assertEqual(loader.around_resolve_quant_load_spec(lambda **kwargs: 7, **self.kwargs), 7)

    def test_non_h3_and_gguf_do_not_inspect_checkpoint(self):
        self.quant.inspect_quant_checkpoint = lambda *args: self.fail("unexpected inspection")
        self.kwargs["model_cls"] = type("OtherDiT", (), {})
        self.assertEqual(loader.around_resolve_quant_load_spec(lambda **kwargs: 1, **self.kwargs), 1)
        self.kwargs["model_cls"] = type("MiniMaxH3DiTModel", (), {})
        self.kwargs["gguf_file"] = "weights.gguf"
        self.assertEqual(loader.around_resolve_quant_load_spec(lambda **kwargs: 2, **self.kwargs), 2)

    def test_fsdp_and_conflicting_quantization_are_rejected_before_original(self):
        def original(**kwargs):
            self.fail("must reject before loading")
        self.args.should_use_fsdp_for_component = lambda component: True
        with self.assertRaisesRegex(ValueError, "FSDP"):
            loader.around_resolve_quant_load_spec(original, **self.kwargs)
        self.args.should_use_fsdp_for_component = lambda component: False
        self.kwargs["checkpoint_quant_config"] = object()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            loader.around_resolve_quant_load_spec(original, **self.kwargs)

    def test_compressed_adaln_rejects_separate_cache(self):
        self.args.minimax_h3_adaln_online = True
        with self.assertRaisesRegex(ValueError, "AdaLN"):
            loader.around_resolve_quant_load_spec(lambda **kwargs: self.fail(), **self.kwargs)

    def test_original_failure_does_not_mutate_architecture(self):
        def original(**kwargs):
            raise RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            loader.around_resolve_quant_load_spec(original, **self.kwargs)
        self.assertTrue(self.arch.checkpoint_uses_diffusers_layout)
        self.assertFalse(hasattr(self.arch, "adaln_curve_grid"))

    def test_official_registry_applies_to_preimported_consumer(self):
        official = Path(os.environ.get("OFFICIAL", str(ROOT.parent / "sglang-official-0.5.22")))
        source = official / "python/sglang/srt/plugins/hook_registry.py"
        if not source.exists():
            self.skipTest("fixed official source is not available")
        spec = importlib.util.spec_from_file_location("h3_contract_registry", source)
        registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry)
        consumer_name = "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader"
        model_name = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
        registry_name = "sglang.srt.plugins.hook_registry"
        modules = {}
        for name in (consumer_name, model_name, registry_name):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                key = ".".join(parts[:i])
                modules.setdefault(key, types.ModuleType(key))
                modules[key].__path__ = []
                if i > 1:
                    setattr(modules[".".join(parts[:i-1])], parts[i-1], modules[key])
        modules[registry_name] = registry
        seen = []
        def original(**kwargs):
            seen.append(kwargs["checkpoint_quant_config"])
            return "actual-plan"
        modules[consumer_name].resolve_transformer_quant_load_spec = original
        modules[consumer_name].inspect_minimax_h3_safetensors = lambda paths: (None, {})
        class Attention:
            def __init__(self):
                self._use_fused_qknorm_rope = True
        modules[model_name].MiniMaxH3Attention = Attention
        with patch.dict(sys.modules, modules):
            loader.register_hooks()
            registry.HookRegistry.apply_hooks()
            self.assertEqual(len(registry.HookRegistry._patched), 3)
            self.assertEqual(modules[consumer_name].resolve_transformer_quant_load_spec(**self.kwargs), "actual-plan")
            self.assertEqual(seen, [self.config])
            self.assertFalse(Attention()._use_fused_qknorm_rope)
        registry.HookRegistry.reset()

    def test_inspector_admits_validated_full_and_compressed_without_comfy_markers(self):
        for profile, curve in (("backbone", None), ("backbone-adaln", (1025, 8))):
            self.quant.inspect_int8_checkpoint = lambda paths: {"profile": profile}
            actual = loader.around_inspect_safetensors(lambda paths: self.fail("legacy format sent to Comfy"), ["weights"])
            self.assertEqual(actual, (curve, {}))

    def test_inspector_preserves_unknown_formats_and_non_hcu(self):
        token = object()
        self.quant.inspect_int8_checkpoint = lambda paths: None
        self.assertIs(loader.around_inspect_safetensors(lambda paths: token, ["weights"]), token)
        loader.is_hcu.return_value = False
        self.quant.inspect_int8_checkpoint = lambda paths: self.fail("unexpected native inspection")
        self.assertIs(loader.around_inspect_safetensors(lambda paths: token, ["weights"]), token)

    def test_invalid_legacy_manifest_remains_an_error(self):
        def invalid(paths):
            raise ValueError("bad manifest")
        self.quant.inspect_int8_checkpoint = invalid
        with self.assertRaisesRegex(ValueError, "bad manifest"):
            loader.around_inspect_safetensors(lambda paths: self.fail("must not downgrade invalid data"), ["weights"])

    def test_real_official_loader_prefix_reaches_quant_plan_after_legacy_inspection(self):
        official = Path(os.environ.get("OFFICIAL", str(ROOT.parent / "sglang-official-0.5.22")))
        path = official / "python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(m, ast.FunctionDef) and m.name == "load_customized" for m in n.body))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_customized")
        prefix = []
        for statement in method.body:
            prefix.append(statement)
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "quant_spec" for t in statement.targets):
                break
        else:
            self.fail("official loader quant plan boundary missing")
        method.body = prefix + [ast.Return(value=ast.Name(id="quant_spec", ctx=ast.Load()))]
        method.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        arch = self.arch
        arch.param_names_mapping, arch.reverse_param_names_mapping = {}, {}
        self.args.model_paths = {}
        self.args.pipeline_config = types.SimpleNamespace(dit_config=types.SimpleNamespace(arch_config=arch, update_model_arch=lambda config: None))
        seen = []
        def legacy_inspector(paths):
            self.fail("old INT8 must not enter Comfy marker inspector")
        def plan(**kwargs):
            seen.append(kwargs["checkpoint_quant_config"])
            return kwargs["checkpoint_quant_config"]
        namespace = dict(
            _server_args_for_transformer_component=lambda args, *a: args,
            get_diffusers_component_config=lambda **kwargs: {"_class_name": "MiniMaxH3DiTModel"},
            resolve_transformer_gguf_to_load=lambda *args: None,
            resolve_transformer_checkpoint_files=lambda *args: types.SimpleNamespace(safetensors=["weights.safetensors"], config_path=None),
            ModelRegistry=types.SimpleNamespace(resolve_model_cls=lambda name: (type("MiniMaxH3DiTModel", (), {}), None)),
            validate_minimax_h3_checkpoint_variant=lambda *args: None,
            inspect_minimax_h3_safetensors=lambda paths: loader.around_inspect_safetensors(legacy_inspector, paths),
            resolve_minimax_h3_checkpoint_quantization=lambda markers, *args: None,
            resolve_transformer_quant_load_spec=lambda **kwargs: loader.around_resolve_quant_load_spec(plan, **kwargs),
        )
        exec(compile(module, str(path), "exec"), namespace)
        instance = types.SimpleNamespace(structural_component_name=lambda name: name, structural_component_type=lambda name: "transformer")
        self.assertIs(namespace["load_customized"](instance, "/weights", self.args, "transformer"), self.config)
        self.assertEqual(seen, [self.config])
        self.assertEqual((arch.adaln_curve_grid, arch.time_embed_dim), (1025, 8))

    def test_hcu_h3_fa_alias_is_canonicalized_before_strict_component_validation(self):
        from hcu_sglang.multimodal_gen import minimax_h3
        name = "sglang.multimodal_gen.runtime.server_args.server_args"
        backend = types.ModuleType(name)
        backend.Backend = types.SimpleNamespace(DIFFUSERS="diffusers")
        args = types.SimpleNamespace(attention_backend="fa", backend="auto", pipeline_config=type("MiniMaxH3PipelineConfig", (), {})(), component_attention_backends={"text_encoder": "torch_sdpa", "transformer": "fa"})
        with patch.dict(sys.modules, {name: backend}), patch.object(minimax_h3, "is_hcu", return_value=True):
            args.component_attention_backend_roles = {}
            args._split_component_attention_backends = lambda components, roles: (dict(components), dict(roles))
            def capture_requested(args):
                args._requested_component_attention_backends = dict(args.component_attention_backends)
            minimax_h3.around_adjust_attention_backend(capture_requested, args)
        self.assertEqual(args._requested_component_attention_backends["transformer"], "fa2")
        self.assertEqual(args.attention_backend, "fa2")
        self.assertEqual(args.component_attention_backends, {"text_encoder": "torch_sdpa", "transformer": "fa2"})

    def test_attention_alias_preserves_other_platforms_and_diffusers(self):
        from hcu_sglang.multimodal_gen import minimax_h3
        name = "sglang.multimodal_gen.runtime.server_args.server_args"
        backend = types.ModuleType(name)
        backend.Backend = types.SimpleNamespace(DIFFUSERS="diffusers")
        for hcu, engine in ((False, "auto"), (True, "diffusers")):
            args = types.SimpleNamespace(attention_backend="fa", backend=engine, pipeline_config=type("MiniMaxH3PipelineConfig", (), {})(), component_attention_backends={"transformer": "fa"})
            with patch.dict(sys.modules, {name: backend}), patch.object(minimax_h3, "is_hcu", return_value=hcu):
                minimax_h3.around_adjust_attention_backend(lambda args: None, args)
            self.assertEqual(args.attention_backend, "fa")
            self.assertEqual(args.component_attention_backends, {"transformer": "fa"})

    def test_attention_gate_preserves_non_hcu(self):
        attention = types.SimpleNamespace(_use_fused_qknorm_rope=True)
        loader.after_attention_init(None, attention)
        self.assertFalse(attention._use_fused_qknorm_rope)
        loader.is_hcu.return_value = False
        attention._use_fused_qknorm_rope = True
        loader.after_attention_init(None, attention)
        self.assertTrue(attention._use_fused_qknorm_rope)


if __name__ == "__main__":
    unittest.main()
