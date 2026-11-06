# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from tools.sync_deepseek_methods import ExtractionError, apply_update, plan_updates


class MethodSyncPreflightTests(unittest.TestCase):
    def test_later_class_drift_does_not_partially_rewrite_outputs(self):
        with tempfile.TemporaryDirectory() as task_dir:
            root = Path(task_dir)
            source = root / "source.py"
            source.write_text("class NewDecoder:\n    def forward(self, x):\n        return x\n")
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("old first\n")
            second.write_text("old second\n")
            jobs = [(first, "", [(source, "NewDecoder", "forward")], {}),
                    (second, "", [(source, "OldDecoder", "forward")], {})]
            with self.assertRaisesRegex(ExtractionError, "available classes: NewDecoder"):
                plan_updates(jobs)
            self.assertEqual(first.read_text(), "old first\n")
            self.assertEqual(second.read_text(), "old second\n")

    def test_check_mode_leaves_existing_output_intact(self):
        with tempfile.TemporaryDirectory() as task_dir:
            output = Path(task_dir) / "methods.py"
            output.write_text("old\n")
            self.assertTrue(apply_update(output, "new\n", check=True))
            self.assertEqual(output.read_text(), "old\n")

    def test_absent_method_reports_exact_symbol(self):
        with tempfile.TemporaryDirectory() as task_dir:
            source = Path(task_dir) / "source.py"
            source.write_text("class Decoder:\n    pass\n")
            with self.assertRaisesRegex(ExtractionError, "Decoder.forward is absent"):
                plan_updates([(source, "", [(source, "Decoder", "forward")], {})])


if __name__ == "__main__":
    unittest.main()

class MethodReferenceSemanticsTests(unittest.TestCase):
    def test_constants_are_identifier_only_and_idempotent(self):
        from tools.sync_deepseek_methods import transform
        source = '''def forward(self):
    # _is_hcu and _is_hcu are explanatory text.
    label = "_is_hcu _is_hcu"
    return _is_hcu, _is_hcu, C._is_hcu, self._is_hcu, label
'''
        mapping = {"_is_hcu": "C._is_hcu", "_is_hcu": "C._is_hcu"}
        first = transform(source, mapping, "Owner")
        self.assertEqual(transform(first, mapping, "Owner"), first)
        namespace = {"C": type("Flags", (), {"_is_hcu": True})}
        exec(first, namespace)
        receiver = type("Receiver", (), {"_is_hcu": "attribute"})()
        self.assertEqual(namespace["forward"](receiver),
                         (True, True, True, "attribute", "_is_hcu _is_hcu"))
        self.assertIn("# _is_hcu and _is_hcu", first)

    def test_super_keeps_parent_semantics_after_detaching(self):
        from tools.sync_deepseek_methods import transform
        class Parent:
            def __init__(self, value):
                self.value = value
        class Owner(Parent):
            pass
        namespace = {"Owner": Owner}
        source = "def __init__(self, value):\n    super().__init__(value + 1)\n"
        transformed = transform(source, {}, "Owner")
        exec(transformed, namespace)
        Owner.__init__ = namespace["__init__"]
        self.assertEqual(Owner(41).value, 42)
        self.assertEqual(transform(transformed, {}, "Owner"), transformed)

    def test_shadowed_constant_is_rejected(self):
        from tools.sync_deepseek_methods import transform
        with self.assertRaisesRegex(ExtractionError, "locally bound"):
            transform("def forward(self, _is_hcu):\n    return _is_hcu\n",
                      {"_is_hcu": "C._is_hcu"}, "Owner")

    def test_unknown_class_owner_is_rejected(self):
        from tools.sync_deepseek_methods import transform
        with self.assertRaisesRegex(ExtractionError, "explicit owner"):
            transform("def __init__(self):\n    super().__init__()\n", {})

    def test_unicode_offsets_preserve_executable_text(self):
        from tools.sync_deepseek_methods import transform
        namespace = {"C": type("Flags", (), {"_is_hcu": True})}
        output = transform('def forward(self):\n    return "海光", _is_hcu\n',
                           {"_is_hcu": "C._is_hcu"}, "Owner")
        exec(output, namespace)
        self.assertEqual(namespace["forward"](None), ("海光", True))

class MethodDecoratorSemanticsTests(unittest.TestCase):
    def test_extracted_decorator_executes_at_call_boundary(self):
        from tools.sync_deepseek_methods import extract_method, transform
        with tempfile.TemporaryDirectory() as task_dir:
            source = Path(task_dir) / 'model.py'
            source.write_text('class Owner:\n    @decorate\n    def forward(self, value):\n        return value\n')
            output = transform(extract_method(source, 'Owner', 'forward'), {}, 'Owner')
            def decorate(fn):
                def wrapper(self, value):
                    return fn(self, value + 1)
                return wrapper
            namespace = {'decorate': decorate}
            exec(output, namespace)
            self.assertEqual(namespace['forward'](None, 41), 42)

    def test_class_descriptor_requires_explicit_extraction_contract(self):
        from tools.sync_deepseek_methods import extract_method
        with tempfile.TemporaryDirectory() as task_dir:
            source = Path(task_dir) / 'model.py'
            source.write_text('class Owner:\n    @classmethod\n    def forward(cls):\n        return cls\n')
            with self.assertRaisesRegex(ExtractionError, 'descriptor'):
                extract_method(source, 'Owner', 'forward')


class ReplacementNamespaceTests(unittest.TestCase):
    def test_provider_and_owner_aliases_cannot_shadow_local_dimensions(self):
        from tools.sync_deepseek_methods import transform
        with self.assertRaisesRegex(ExtractionError, "Replacement names are locally bound: N"):
            transform("def forward(self, x):\n    N = 2\n    return helper(x) + N\n", {"helper": "N.helper"})
        with self.assertRaisesRegex(ExtractionError, "Replacement names are locally bound: N"):
            transform("def __init__(self):\n    N = 2\n    super().__init__()\n", {}, "N.Owner")

    def test_distinct_provider_alias_executes_and_remains_idempotent(self):
        from types import SimpleNamespace
        from tools.sync_deepseek_methods import transform
        source = "def forward(self, x):\n    N = 2\n    return helper(x) + N\n"
        mapped = transform(source, {"helper": "native_provider.helper"})
        namespace = {"native_provider": SimpleNamespace(helper=lambda value: value * 3)}
        exec(mapped, namespace)
        self.assertEqual(namespace["forward"](None, 7), 23)
        self.assertEqual(transform(mapped, {"helper": "native_provider.helper"}), mapped)


class CurrentV4ProfileTests(unittest.TestCase):
    def make_tree(self, root):
        from tools.sync_deepseek_methods import CURRENT_V4_METHODS
        source = root / "source"
        models = root / "hcu_sglang/models"
        owned = models / "deepseek/current_v4/execution/core.py"
        source.mkdir()
        owned.parent.mkdir(parents=True)
        content = "class DeepseekV4DecoderLayerBase: pass\n"
        content += "class DeepseekV4PlatformDecoderLayer(NonCudaDecoderMixin, DeepseekV4DecoderLayerBase): pass\n"
        for class_name, names in CURRENT_V4_METHODS.values():
            content += f"class {class_name}:\n"
            content += "".join(f"    def {name}(self): return super().forward()\n" for name in names)
        (source / "deepseek_v4.py").write_text(content)
        owned.write_text(content)
        return source, models, owned

    def test_profile_preserves_class_owner_and_only_plans_facades(self):
        from tools.sync_deepseek_methods import plan_current_v4_updates
        with tempfile.TemporaryDirectory() as task_dir:
            source, models, owned = self.make_tree(Path(task_dir))
            before = owned.read_bytes()
            updates = plan_current_v4_updates(source, models, "hcu_sglang")
            self.assertEqual(len(updates), 2)
            self.assertEqual(owned.read_bytes(), before)
            self.assertTrue(all(path.parent == models / "deepseek/deepseek_v4/methods" for path, _ in updates))
            self.assertIn("_forward_impl = C.DeepseekV4Model._forward_impl", updates[0][1])
            self.assertNotIn("super(", updates[0][1])

    def test_source_class_split_drift_fails_before_any_write(self):
        from tools.sync_deepseek_methods import plan_current_v4_updates
        with tempfile.TemporaryDirectory() as task_dir:
            source, models, owned = self.make_tree(Path(task_dir))
            path = source / "deepseek_v4.py"
            path.write_text(path.read_text().replace("(NonCudaDecoderMixin, DeepseekV4DecoderLayerBase)", "(DeepseekV4DecoderLayerBase, NonCudaDecoderMixin)"))
            before = owned.read_bytes()
            with self.assertRaisesRegex(ExtractionError, "Platform decoder MRO"):
                plan_current_v4_updates(source, models, "hcu_sglang")
            self.assertEqual(owned.read_bytes(), before)
            self.assertFalse((models / "deepseek/deepseek_v4/methods").exists())

    def test_owned_missing_forward_body_fails_before_any_write(self):
        from tools.sync_deepseek_methods import plan_current_v4_updates
        with tempfile.TemporaryDirectory() as task_dir:
            source, models, owned = self.make_tree(Path(task_dir))
            owned.write_text(owned.read_text().replace("def _forward_impl(", "def other("))
            with self.assertRaisesRegex(ExtractionError, "DeepseekV4Model._forward_impl is absent"):
                plan_current_v4_updates(source, models, "hcu_sglang")
            self.assertFalse((models / "deepseek/deepseek_v4/methods").exists())
