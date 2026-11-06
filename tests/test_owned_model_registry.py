from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.models import registry as R

ROOT = Path(__file__).resolve().parents[1]


class OwnedModelSourceTests(unittest.TestCase):
    def test_owned_sources_preserve_snapshot_except_declared_imports(self):
        manifest = json.loads((ROOT / "tests/contracts/owned_model_sources.json").read_text())
        self.assertEqual(len(manifest["files"]), 6)
        for row in manifest["files"]:
            with self.subTest(source=row["source"]):
                text = (ROOT / row["owned"]).read_text()
                compile(text, row["owned"], "exec")
                for source, owned in row["replacements"].items():
                    self.assertIn(owned, text)
                    text = text.replace(owned, source)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), row["source_sha256"])

    def test_entry_modules_declare_exact_owned_architectures(self):
        for family, (package, prefix, expected) in R._FAMILIES.items():
            with self.subTest(family=family):
                found = set()
                for entry in (ROOT / package.replace(".", "/")).glob("*.py"):
                    tree = ast.parse(entry.read_text(), filename=str(entry))
                    imported = {
                        alias.asname or alias.name: (node.module, alias.name)
                        for node in tree.body if isinstance(node, ast.ImportFrom)
                        for alias in node.names
                    }
                    for node in tree.body:
                        if not isinstance(node, ast.Assign):
                            continue
                        if not any(isinstance(target, ast.Name) and target.id == "EntryClass" for target in node.targets):
                            continue
                        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                        for value in values:
                            module, name = imported[value.id]
                            self.assertTrue(module.startswith(prefix), (module, name))
                            self.assertEqual(value.id, name)
                            found.add(name)
                self.assertEqual(found, expected)


class OwnedModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.detection = ModuleType("hcu_sglang.utils.detection")
        self.detection.is_hcu = Mock(return_value=True)
        self.modules = patch.dict(sys.modules, {self.detection.__name__: self.detection})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.registry = SimpleNamespace(models={})
        self.registry.register = Mock(side_effect=self.install_family)
        self.original = Mock(return_value=(object(), "resolved"))

    def install_family(self, package, *, overwrite, strict):
        for candidate, prefix, names in R._FAMILIES.values():
            if candidate == package:
                self.registry.models.update({name: type(name, (), {"__module__": prefix + "model"}) for name in names})

    def resolve(self, architectures):
        return R.around_resolve_model_cls(self.original, self.registry, architectures)

    def test_glm_resolution_uses_strict_formal_registry_and_is_idempotent(self):
        result = self.resolve("Glm5NextForCausalLM")
        self.assertIs(result, self.original.return_value)
        self.registry.register.assert_called_once_with("hcu_sglang.models.glm5_next.entries", overwrite=True, strict=True)
        self.resolve(["Glm5NextForConditionalGenerationNextN"])
        self.assertEqual(self.registry.register.call_count, 1)
        self.original.assert_called_with(self.registry, ["Glm5NextForConditionalGenerationNextN"])

    def test_muse_and_glm_load_only_the_requested_family(self):
        for name, family in [("MuseSparkForConditionalGeneration", "muse_spark"), ("Glm5NextForCausalLMNextN", "glm5_next")]:
            with self.subTest(architecture=name):
                self.registry.models.clear()
                self.registry.register.reset_mock()
                self.resolve([name])
                self.registry.register.assert_called_once_with(R._FAMILIES[family][0], overwrite=True, strict=True)

    def test_non_hcu_and_unrelated_models_do_not_load_owned_families(self):
        self.detection.is_hcu.return_value = False
        self.resolve("Glm5NextForCausalLM")
        self.detection.is_hcu.return_value = True
        self.resolve(["Qwen3ForCausalLM"])
        self.registry.register.assert_not_called()
        self.assertEqual(self.original.call_count, 2)

    def test_missing_dependency_propagates_and_does_not_fall_back(self):
        error = ModuleNotFoundError("GLM communicator dependency missing")
        self.registry.register.side_effect = error
        with self.assertRaises(ModuleNotFoundError) as raised:
            self.resolve(["Glm5NextForCausalLM"])
        self.assertIs(raised.exception, error)
        self.original.assert_not_called()
        self.registry.register.side_effect = self.install_family
        self.resolve("Glm5NextForCausalLM")
        self.assertEqual(self.registry.register.call_count, 2)

    def test_disabled_or_missing_entry_cannot_select_official_implementation(self):
        self.registry.models["Glm5NextForCausalLM"] = type("Glm5NextForCausalLM", (), {"__module__": "sglang.srt.models.glm5_next"})
        self.registry.register.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "was not registered"):
            self.resolve("Glm5NextForCausalLM")
        self.original.assert_not_called()

    def test_registration_uses_public_hook_and_keeps_non_hcu_untouched(self):
        hooks = ModuleType("sglang.srt.plugins.hook_registry")
        hooks.HookRegistry = SimpleNamespace(register=Mock())
        hooks.HookType = SimpleNamespace(AROUND=object())
        with patch.dict(sys.modules, {hooks.__name__: hooks}):
            R.register_hooks()
            hooks.HookRegistry.register.assert_called_once_with("sglang.srt.models.registry._ModelRegistry.resolve_model_cls", R.around_resolve_model_cls, hooks.HookType.AROUND)
            hooks.HookRegistry.register.reset_mock()
            self.detection.is_hcu.return_value = False
            R.register_hooks()
            hooks.HookRegistry.register.assert_not_called()


class OwnedGlmNextNTests(unittest.TestCase):
    def setUp(self):
        parent = ModuleType("sglang.srt.models.deepseek_nextn")
        self.parent_init = Mock()
        self.quant_fallback = object()
        test = self

        class Parent:
            def __init__(self, *args, **kwargs):
                test.parent_init(*args, **kwargs)

            def _resolve_nextn_quant_config(self, config, quant_config):
                return test.quant_fallback

        parent.DeepseekV3ForCausalLMNextN = Parent
        model = ModuleType("hcu_sglang.models.glm5_next.model")
        self.loaded = []

        def load_weights(owner, weights, *, is_nextn):
            self.assertTrue(is_nextn)
            self.loaded.extend(weights)
            return self.loaded

        model.Glm5NextForCausalLM = SimpleNamespace(load_weights=load_weights)
        utils = ModuleType("sglang.srt.models.utils")
        utils.WeightsMapper = lambda **kwargs: SimpleNamespace(**kwargs)
        self.module = ModuleType("owned_nextn_cpu_fixture")
        path = ROOT / "hcu_sglang/models/glm5_next/nextn.py"
        with patch.dict(sys.modules, {parent.__name__: parent, model.__name__: model, utils.__name__: utils}):
            exec(compile(path.read_text(), str(path), "exec"), self.module.__dict__)
        self.cls = self.module.Glm5NextForConditionalGenerationNextN

    def test_mapper_uses_target_layer_count_for_composite_and_text_config(self):
        text = SimpleNamespace(num_hidden_layers=7)
        for config in [text, SimpleNamespace(text_config=text)]:
            self.assertEqual(self.cls.get_hf_to_sglang_mapper(config).orig_to_new_substr, {"model.layers.7": "model.decoder"})

    def test_loader_filters_target_layer_and_preserves_tensor_identity(self):
        owner = object.__new__(self.cls)
        owner.config = SimpleNamespace(num_hidden_layers=7, q_lora_rank=4)
        a, b = object(), object()
        result = owner.load_weights([
            ("model.layers.7.self_attn.q_proj.weight", a),
            ("model.language_model.layers.7.mlp.gate_proj.weight", b),
            ("model.layers.70.self_attn.q_proj.weight", object()),
            ("model.layers.6.mlp.gate_proj.weight", object()),
            ("visual.proj.weight", object()),
        ])
        self.assertIs(result, self.loaded)
        self.assertEqual(self.loaded, [("model.layers.7.self_attn.q_proj.weight", a), ("model.layers.7.mlp.gate_proj.weight", b)])
        self.assertTrue(owner.fuse_qkv_a_proj)

    def test_declared_bf16_nextn_layer_disables_global_quantization(self):
        owner = object.__new__(self.cls)
        config = SimpleNamespace(num_hidden_layers=7, quantization_config={"ignore": ["model.layers.7.*"]})
        self.assertIsNone(owner._resolve_nextn_quant_config(config, object()))
        config.quantization_config = {"ignore": ["model.layers.6.*"]}
        self.assertIs(owner._resolve_nextn_quant_config(config, object()), self.quant_fallback)

    def test_cp_handoff_embeds_full_batch_before_runner_shards(self):
        owner = object.__new__(self.cls)
        embeddings, ids, positions = object(), object(), object()
        owner.model = SimpleNamespace(embed_input_ids=Mock(return_value=embeddings))
        batch = SimpleNamespace(input_ids=ids, positions=positions)
        self.assertEqual(owner.prepare_cp_inputs(batch), (embeddings, positions, {}))
        owner.model.embed_input_ids.assert_called_once_with(ids, batch)

    def test_text_and_composite_constructor_delegate_to_same_parent(self):
        text, quant = object(), object()
        self.cls(SimpleNamespace(text_config=text), quant_config=quant, prefix="draft")
        self.parent_init.assert_called_once_with(text, quant_config=quant, prefix="draft")
        self.assertTrue(issubclass(self.module.Glm5NextForCausalLMNextN, self.cls))


if __name__ == "__main__":
    unittest.main()
