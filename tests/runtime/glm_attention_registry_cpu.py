from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hcu_sglang
from hcu_sglang.layers.attention import glm5_next_registry as registry
from sglang.srt.plugins.hook_registry import HookRegistry


class GlmAttentionRegistryTests(unittest.TestCase):
    def test_glm_uses_owned_native_sparse_backend_and_other_models_fall_back(self):
        hcu_sglang.register()
        HookRegistry.apply_hooks()
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="glm5_next")
            )
        )
        selected = object()
        with (
            patch.object(registry, "is_hcu", return_value=True),
            patch.object(registry, "is_glm5_next_hcu", return_value=True),
            patch(
                "hcu_sglang.layers.attention.glm5_next.dsa_backend.NativeSparseAttnBackend",
                return_value=selected,
            ) as backend,
        ):
            self.assertIs(registry.around_create_dsa_backend(Mock(), runner), selected)
            backend.assert_called_once_with(runner)
        original = Mock(return_value="native")
        with (
            patch.object(registry, "is_hcu", return_value=True),
            patch.object(registry, "is_glm5_next_hcu", return_value=False),
        ):
            self.assertEqual(registry.around_create_dsa_backend(original, runner), "native")
        original.assert_called_once_with(runner)


if __name__ == "__main__":
    unittest.main()
