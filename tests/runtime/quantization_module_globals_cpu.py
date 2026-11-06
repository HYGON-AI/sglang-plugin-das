from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hcu_sglang.layers.quantization import quark_w8a8_fp8_moe as quark
from hcu_sglang.layers.quantization import registry


class QuantizationModuleTests(unittest.TestCase):
    def test_hcu_quark_disables_shuffle_without_changing_other_platforms(self):
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat

        bootstrap_aiter_compat()
        native = importlib.import_module(
            "sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8_moe"
        )
        original = getattr(native, "_is_hcu", None), native._use_aiter
        self.addCleanup(setattr, native, "_is_hcu", original[0])
        self.addCleanup(setattr, native, "_use_aiter", original[1])
        with patch.object(quark, "is_hcu", return_value=True):
            quark.register_hooks()
            self.assertTrue(native._is_hcu)
            self.assertFalse(native._use_aiter)
        native._use_aiter = True
        with patch.object(quark, "is_hcu", return_value=False):
            quark.register_hooks()
            self.assertFalse(native._is_hcu)
            self.assertTrue(native._use_aiter)

    def test_quantization_registry_adds_source_names_without_overriding_existing(self):
        existing = object()
        quark_config = type("QuarkConfig", (), {})
        quantization = SimpleNamespace(
            QuarkConfig=quark_config,
            BASE_QUANTIZATION_METHODS={"quark": existing},
            QUANTIZATION_METHODS={},
        )
        model_config = SimpleNamespace(QUANTIZATION_METHODS={})
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.quantization": quantization,
                "sglang.srt.configs.model_config": model_config,
            },
        ):
            self.assertTrue(registry.install_quantization_method())
        self.assertIs(quantization.BASE_QUANTIZATION_METHODS["quark"], existing)
        self.assertIs(
            quantization.BASE_QUANTIZATION_METHODS["quark_mxfp4"], quark_config
        )
        self.assertIs(quantization.QUANTIZATION_METHODS["quark"], quark_config)
        self.assertIn("slimquant_w4a8", quantization.BASE_QUANTIZATION_METHODS)
        self.assertIn("slimquant_w4a8_marlin", model_config.QUANTIZATION_METHODS)


if __name__ == "__main__":
    unittest.main()
