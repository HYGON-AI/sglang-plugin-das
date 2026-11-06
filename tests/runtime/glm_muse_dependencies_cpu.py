from __future__ import annotations

import importlib
import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch

import hcu_sglang


class GlmMuseDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hcu_sglang.register()
        cls.cp = importlib.import_module("hcu_sglang.layers.glm5_next_cp_utils")
        cls.mhc = importlib.import_module(
            "hcu_sglang.layers.communicator_glm5_next_mhc"
        )
        cls.mhc_cp = importlib.import_module(
            "hcu_sglang.layers.communicator_glm5_next_mhc_cp"
        )

    def test_plain_cp_helpers_cover_uneven_shapes_and_single_rank_identity(self):
        self.assertEqual(self.cp._plain_cp_chunk_sizes(10, 4), [3, 3, 2, 2])
        value = torch.arange(12).reshape(6, 2)
        parallel = type("Parallel", (), {"attn_cp_rank": 0})()
        with patch.object(self.cp, "get_parallel", return_value=parallel):
            self.assertIs(self.cp.cp_plain_split(value, 1), value)
        self.assertIs(self.cp.cp_plain_all_gather(value, 1), value)
        self.assertIs(self.cp.cp_plain_reduce_scatter(value, 1), value)

    def test_mhc_prepare_attn_preserves_cross_layer_state_producers(self):
        owner = object.__new__(self.mhc.MHCLayerCommunicator)
        owner.is_first_layer = False
        owner.qkv_latent_func = None
        owner.input_layernorm = type(
            "Norm",
            (),
            {
                "weight": torch.ones(2),
                "variance_epsilon": 0.0,
                "__call__": lambda self, value: value + 1,
            },
        )()
        h_res, h_post = object(), object()

        def produce(value, weight, epsilon):
            self.assertIs(weight, owner.input_layernorm.weight)
            self.assertEqual(epsilon, 0.0)
            return value * 2, h_res, h_post, False

        owner.hc_attn_pre = produce
        value = torch.tensor([[1.0, 2.0]])
        actual, residual = owner.prepare_attn(value, None, object())
        torch.testing.assert_close(actual, value * 2 + 1)
        self.assertIs(residual, value)
        self.assertIs(owner._h_res, h_res)
        self.assertIs(owner._h_post, h_post)
        self.assertIsNone(owner._attn_input_quant_args)

    def test_mhc_cp_fallback_pairs_gather_scatter_and_clears_state(self):
        owner = object.__new__(self.mhc_cp.Glm5NextMHCCPLayerCommunicator)
        owner.is_layer_sparse = True
        value = torch.tensor([[1.0, 2.0]])
        backend = Mock()
        backend.is_none.return_value = True
        with (
            patch.object(self.mhc_cp, "dsa_use_prefill_cp", return_value=True),
            patch.object(self.mhc_cp, "get_moe_a2a_backend", return_value=backend),
            patch.object(
                self.mhc_cp, "cp_plain_all_gather", side_effect=lambda x, **kwargs: x * 2
            ) as gather,
            patch.object(
                self.mhc_cp, "cp_plain_reduce_scatter", side_effect=lambda x: x + 1
            ) as scatter,
        ):
            gathered = owner.prepare_mlp_input(value, object(), True)
            restored = owner.restore_mlp_output(gathered, object())
        torch.testing.assert_close(gathered, value * 2)
        torch.testing.assert_close(restored, value * 2 + 1)
        gather.assert_called_once()
        scatter.assert_called_once()
        self.assertIsNone(owner._mlp_comm_kind)
        self.assertIsNone(owner._a2a_scatter_chunks)

    def test_muse_helion_adapter_preserves_dispatch_and_cpu_rejection(self):
        spec = importlib.import_module("hcu_sglang.kernels.muse_spark_spec")
        module = importlib.import_module(
            "hcu_sglang.kernels.ops.muse_spark.fused_norm_gate"
        )
        from sglang.kernels import fused_op

        self.assertEqual(
            fused_op.BACKEND_METHODS[spec.KernelBackend.HELION],
            "forward_helion",
        )
        op = module.FusedRMSNormGate(None, False, 128)
        with patch.object(
            fused_op,
            "_platform",
            return_value=spec.PlatformInfo(device_type="cpu"),
        ):
            self.assertFalse(
                op.backend_eligible(
                    spec.KernelBackend.HELION, torch.empty(1, 128)
                )
            )
            self.assertIsNone(op.auto_selected_backend())

    def test_optional_helion_modules_report_missing_dependency(self):
        if importlib.util.find_spec("helion") is not None:
            self.skipTest("optional Helion package is installed")
        for name in (
            "helion_norm_fp4_quant",
            "helion_norm_gate",
            "helion_post_expert",
            "helion_qk_norm",
        ):
            with self.subTest(module=name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(
                        f"hcu_sglang.kernels.ops.muse_spark.{name}"
                    )


if __name__ == "__main__":
    unittest.main()
