from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import dump_runtime_hook_targets as dump


class OwnerCaptureTests(unittest.TestCase):
    def test_owner_records_module_name_and_file(self):
        module = types.ModuleType("hcu_sglang_probe_module")
        module.__file__ = "/tmp/hcu_sglang_probe_module.py"
        sys.modules["hcu_sglang_probe_module"] = module

        def hook():
            return None

        hook.__module__ = "hcu_sglang_probe_module"
        record = dump.owner_of(hook)
        self.assertEqual(record["module"], "hcu_sglang_probe_module")
        self.assertEqual(record["file"], "/tmp/hcu_sglang_probe_module.py")
        self.assertEqual(record["name"], "OwnerCaptureTests.test_owner_records_module_name_and_file.<locals>.hook")

    def test_owner_with_unloaded_module_records_empty_file(self):
        def hook():
            return None

        hook.__module__ = "hcu_sglang_module_that_is_not_loaded"
        record = dump.owner_of(hook)
        self.assertEqual(record["module"], "hcu_sglang_module_that_is_not_loaded")
        self.assertEqual(record["file"], "")


if __name__ == "__main__":
    unittest.main()
