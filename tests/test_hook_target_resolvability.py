from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import scan_hook_targets as scan


class ResolverGateTests(unittest.TestCase):
    """Two gates that any full-scan number must pass: known present resolves, known absent is reported."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        (root / "sglang/probe").mkdir(parents=True)
        (root / "sglang/__init__.py").write_text("")
        (root / "sglang/probe/__init__.py").write_text("")
        (root / "sglang/probe/base.py").write_text(
            "class Base:\n"
            "    def inherited_method(self):\n"
            "        return 1\n"
        )
        (root / "sglang/probe/targets.py").write_text(
            "from sglang.probe.base import Base\n"
            "\n"
            "if True:\n"
            "    def defined_in_conditional_block():\n"
            "        return 1\n"
            "\n"
            "class Derived(Base):\n"
            "    def own_method(self):\n"
            "        return 1\n"
        )
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def resolve(self, targets):
        return scan.check_resolvable(targets, self.root, {}, minimum_files=1)

    def test_known_present_definitions_resolve(self):
        targets = {
            "sglang.probe.targets.defined_in_conditional_block": "hcu_sglang/probe.py",
            "sglang.probe.targets.Derived.own_method": "hcu_sglang/probe.py",
            "sglang.probe.targets.Derived.inherited_method": "hcu_sglang/probe.py",
            "sglang.probe.base.Base.inherited_method": "hcu_sglang/probe.py",
        }
        resolvable, unresolvable, undetermined = self.resolve(targets)
        self.assertEqual(sorted(resolvable), sorted(targets))
        self.assertEqual(unresolvable, [])
        self.assertEqual(undetermined, [])

    def test_known_absent_definitions_are_reported_missing(self):
        targets = {
            "sglang.probe.targets.missing_function": "hcu_sglang/probe.py",
            "sglang.probe.targets.Derived.missing_method": "hcu_sglang/probe.py",
        }
        resolvable, unresolvable, undetermined = self.resolve(targets)
        self.assertEqual(resolvable, [])
        self.assertEqual(undetermined, [])
        self.assertEqual(
            sorted(row["reason"] for row in unresolvable),
            ["missing_class_attribute", "missing_module_attribute"],
        )

    def test_missing_submodule_is_distinguished_from_missing_attribute(self):
        targets = {"sglang.absent.module.symbol": "hcu_sglang/probe.py"}
        resolvable, unresolvable, undetermined = self.resolve(targets)
        self.assertEqual(resolvable, [])
        self.assertEqual(undetermined, [])
        self.assertEqual([row["reason"] for row in unresolvable], ["missing_submodule"])


    def test_attribute_installed_by_plugin_before_register_is_not_a_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = root / "hcu_sglang/probe_hooks.py"
            owner.parent.mkdir(parents=True)
            owner.write_text(
                "def register_hooks(native):\n"
                "    if not hasattr(native, 'write_mtp_seed'):\n"
                "        native.write_mtp_seed = None\n"
                "    HookRegistry.register('sglang.probe.targets.write_mtp_seed', None, None)\n"
            )
            targets = {"sglang.probe.targets.write_mtp_seed": "hcu_sglang/probe_hooks.py"}
            resolvable, unresolvable, undetermined = scan.check_resolvable(
                targets, self.root, {}, minimum_files=1, owner_root=root
            )
        self.assertEqual(resolvable, [])
        self.assertEqual(unresolvable, [])
        self.assertEqual([row["reason"] for row in undetermined], ["installed_by_plugin_before_register"])


class OfficialTreeResolverTests(unittest.TestCase):
    """Regression against the real official tree for the two failures this resolver produced."""

    OFFICIAL = ROOT.parent / "input-code/sglang/python"

    def setUp(self):
        if not self.OFFICIAL.is_dir():
            self.skipTest("official sglang tree is not next to the plugin checkout")

    def test_definition_inside_conditional_block_resolves(self):
        targets = {"sglang.kernels.ops.quantization.fp8_kernel.scaled_fp8_quant": "hcu_sglang/layers/quantization/fp8_quant.py"}
        resolvable, unresolvable, undetermined = scan.check_resolvable(targets, self.OFFICIAL, {})
        self.assertEqual(resolvable, list(targets))
        self.assertEqual(unresolvable, [])
        self.assertEqual(undetermined, [])

    def test_das_only_module_symbol_is_reported_missing(self):
        targets = {"sglang.srt.layers.cp.utils.enable_cp_v2": "hcu_sglang/cp_runtime.py"}
        resolvable, unresolvable, undetermined = scan.check_resolvable(targets, self.OFFICIAL, {})
        self.assertEqual(resolvable, [])
        self.assertEqual(undetermined, [])
        self.assertEqual(len(unresolvable), 1)
        self.assertEqual(unresolvable[0]["reason"], "missing_module_attribute")


if __name__ == "__main__":
    unittest.main()
