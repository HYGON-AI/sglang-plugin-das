# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
from hcu_sglang.environ import register_envs

bootstrap_aiter_compat()
register_envs()

from hcu_sglang.layers.moe import deepep_fp8 as hooks
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.utils import DeepEPMode


class DeepEPW4A8Tests(unittest.TestCase):
    def test_masked_moe_reaches_the_hcu_gemms(self):
        import deepgemm

        dispatch = hooks.D.DeepEPLLDispatchOutput(
            torch.ones(2, 4, 8, dtype=torch.int8),
            torch.ones(2, 4, 1),
            torch.tensor([[0, -1]], dtype=torch.int64),
            torch.tensor([[0.75, 0.0]]),
            torch.tensor([2, 0], dtype=torch.int32),
            6,
        )
        layer = SimpleNamespace(
            deprecate_flag=False,
            use_w4afp8=False,
            _hcu_slimquant_w4a8_hipc=True,
            moe_runner_config=SimpleNamespace(activation="silu", swiglu_limit=None),
            w13_weight=torch.empty(2, 8, 4, dtype=torch.int8),
            w13_weight_scale=torch.ones(2, 8, 1),
            w2_weight=torch.empty(2, 8, 2, dtype=torch.int8),
            w2_weight_scale=torch.ones(2, 8, 1),
        )

        def gemm(a, b, out, mask, expected):
            self.assertIs(mask, dispatch.masked_m)
            self.assertEqual(expected, 4)
            out.fill_(2)

        a2 = (torch.ones(2, 4, 4, dtype=torch.int8), torch.ones(2, 4, 1))
        with (
            patch.object(hooks, "_is_hcu", True),
            patch.object(deepgemm, "m_grouped_w4a8_gemm_nt_masked_hipc", side_effect=gemm) as op,
            patch("lightop.activation.fuse_silu_mul_quant_ep", return_value=a2),
        ):
            combined = hooks.around_run_moe_core(DeepEPMoE.run_moe_core, layer, dispatch)

        self.assertEqual(op.call_count, 2)
        self.assertIs(op.call_args_list[0].args[0][0], dispatch.hidden_states)
        self.assertIs(op.call_args_list[1].args[0], a2)
        self.assertIs(combined.topk_ids, dispatch.topk_ids)
        self.assertIs(combined.topk_weights, dispatch.topk_weights)
        self.assertEqual(combined.hidden_states.dtype, torch.bfloat16)


    def test_masked_dispatch_config_leaves_normal_dispatch_untouched(self):
        from hcu_sglang.layers.moe import deepep_current, deepep_w4a8

        dispatcher = object.__new__(deepep_current.DeepEPDispatcher)
        dispatcher.deepep_mode = DeepEPMode.AUTO
        dispatcher.quant_config = {"existing": True}
        dispatcher._normal_dispatcher = Mock()
        ll = object.__new__(deepep_current._DeepEPDispatcherImplLowLatency)
        ll.set_deepep_dispatcher_dtype = Mock()
        dispatcher._low_latency_dispatcher = ll
        layer = SimpleNamespace(
            dispatcher=dispatcher, moe_runner_config=SimpleNamespace(activation="silu")
        )

        with (
            patch.object(deepep_current, "_is_hcu", True),
            patch.object(deepep_current, "use_groupgemm", True),
            patch.object(deepep_current, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", True),
        ):
            deepep_w4a8.configure_hipc(layer, masked=True, contiguous=False)

        self.assertTrue(layer._hcu_slimquant_w4a8_hipc)
        self.assertTrue(ll.quant_config["hcu_w4a8_int8_dispatch"])
        self.assertTrue(ll.quant_config["hcu_legacy_groupgemm"])
        self.assertEqual(ll.quant_config["low_latency_dispatcher_output_dtype"], "int8")
        normal_config = dispatcher._normal_dispatcher.set_quant_config.call_args.args[0]
        self.assertTrue(normal_config["existing"])
        self.assertNotIn("hcu_w4a8_int8_dispatch", normal_config)
        self.assertNotIn("low_latency_dispatcher_output_dtype", normal_config)

    def test_tbo_wrapper_unwraps_to_the_hcu_dispatcher(self):
        from hcu_sglang.layers.moe import deepep_current, deepep_w4a8

        def make_inner():
            dispatcher = object.__new__(deepep_current.DeepEPDispatcher)
            dispatcher.deepep_mode = DeepEPMode.AUTO
            dispatcher.quant_config = {"existing": True}
            dispatcher._normal_dispatcher = Mock()
            ll = object.__new__(deepep_current._DeepEPDispatcherImplLowLatency)
            ll.set_deepep_dispatcher_dtype = Mock()
            dispatcher._low_latency_dispatcher = ll
            return dispatcher, ll

        inners = [make_inner(), make_inner()]

        class TboWrapper:
            def __init__(self, inner_dispatchers):
                self._inners = inner_dispatchers
                self.quant_config = {}

            def set_quant_config(self, quant_config):
                self.quant_config = quant_config
                for inner in self._inners:
                    inner.set_quant_config(quant_config)

        layer = SimpleNamespace(
            dispatcher=TboWrapper([inner for inner, _ in inners]),
            moe_runner_config=SimpleNamespace(activation="silu"),
        )

        with (
            patch.object(deepep_current, "_is_hcu", True),
            patch.object(deepep_current, "use_groupgemm", True),
            patch.object(deepep_current, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", True),
        ):
            deepep_w4a8.configure_hipc(layer, masked=True, contiguous=False)

        self.assertTrue(layer._hcu_slimquant_w4a8_hipc)
        for _, ll in inners:
            self.assertTrue(ll.quant_config["hcu_w4a8_int8_dispatch"])
            self.assertTrue(ll.quant_config["existing"])

    def test_requested_mode_missing_from_deepep_mode_is_rejected(self):
        from hcu_sglang.layers.moe import deepep_current, deepep_w4a8

        for deepep_mode, kwargs, expected in (
            (DeepEPMode.NORMAL, dict(masked=True, contiguous=False), "MASKED_HIPC"),
            (DeepEPMode.LOW_LATENCY, dict(masked=False, contiguous=True), "CONTIGUOUS_HIPC"),
        ):
            dispatcher = object.__new__(deepep_current.DeepEPDispatcher)
            dispatcher.deepep_mode = deepep_mode
            dispatcher.quant_config = {}
            dispatcher.set_quant_config = Mock()
            layer = SimpleNamespace(
                dispatcher=dispatcher,
                moe_runner_config=SimpleNamespace(activation="silu"),
            )
            with (
                self.subTest(deepep_mode=deepep_mode),
                patch.object(deepep_current, "_is_hcu", True),
                patch.object(deepep_current, "use_groupgemm", True),
                patch.object(deepep_current, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", True),
                self.assertRaisesRegex(RuntimeError, expected),
            ):
                deepep_w4a8.configure_hipc(layer, **kwargs)

            dispatcher.set_quant_config.assert_not_called()
            self.assertFalse(getattr(layer, "_hcu_slimquant_w4a8_hipc", False))
            self.assertFalse(getattr(layer, "_hcu_slimquant_w4a8_contiguous_hipc", False))

    def test_non_hcu_dispatcher_is_rejected(self):
        from hcu_sglang.layers.moe import deepep_current, deepep_w4a8

        layer = SimpleNamespace(
            dispatcher=SimpleNamespace(_inners=[SimpleNamespace()]),
            moe_runner_config=SimpleNamespace(activation="silu"),
        )
        with (
            patch.object(deepep_current, "_is_hcu", True),
            self.assertRaisesRegex(RuntimeError, "requires the HCU DeepEP dispatcher"),
        ):
            deepep_w4a8.configure_hipc(layer, masked=True, contiguous=False)

    def test_abi_error_names_the_failing_term(self):
        from hcu_sglang.layers.moe import deepep_current

        vendor_buffer = SimpleNamespace(
            low_latency_dispatch=lambda x, quant_type=1: x
        )
        upstream_buffer = SimpleNamespace(low_latency_dispatch=lambda x, use_fp8=True: x)

        def adapter_bridged(x, use_fp8=False, **unmapped):
            return x

        adapter_bridged.__module__ = "hcu_adapter.bridges.sglang"
        bridged_buffer = SimpleNamespace(low_latency_dispatch=adapter_bridged)

        cases = (
            (
                dict(_is_hcu=False, use_deepep=True, Buffer=vendor_buffer),
                "not detected as an HCU",
            ),
            (
                dict(_is_hcu=True, use_deepep=False, Buffer=vendor_buffer),
                "failed to import",
            ),
            (
                dict(_is_hcu=True, use_deepep=True, Buffer=upstream_buffer),
                "upstream use_fp8 dispatch ABI",
            ),
            (
                dict(_is_hcu=True, use_deepep=True, Buffer=bridged_buffer),
                "set HCU_DEEPEP_BRIDGE=0",
            ),
            (dict(_is_hcu=True, use_deepep=True, Buffer=vendor_buffer), None),
        )
        for kwargs, expected in cases:
            with self.subTest(expected=expected), patch.multiple(
                deepep_current, _deepep_import_error=ImportError("boom"), **kwargs
            ):
                reason = deepep_current.deepep_ll_dispatch_abi_error()
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)

    def test_masked_hipc_missing_abi_names_the_wheel(self):
        from hcu_sglang.layers.moe import deepep_current, deepep_w4a8

        dispatcher = object.__new__(deepep_current.DeepEPDispatcher)
        dispatcher.deepep_mode = DeepEPMode.AUTO
        dispatcher.quant_config = {}
        dispatcher.set_quant_config = Mock()
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            moe_runner_config=SimpleNamespace(activation="silu"),
        )
        upstream_buffer = SimpleNamespace(low_latency_dispatch=lambda x, use_fp8=True: x)

        with (
            patch.object(deepep_current, "_is_hcu", True),
            patch.object(deepep_current, "use_deepep", True),
            patch.object(deepep_current, "use_groupgemm", True),
            patch.object(deepep_current, "Buffer", upstream_buffer),
            patch.object(deepep_current, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", False),
            self.assertRaisesRegex(RuntimeError, "upstream use_fp8 dispatch ABI"),
        ):
            deepep_w4a8.configure_hipc(layer, masked=True, contiguous=False)

        dispatcher.set_quant_config.assert_not_called()
        self.assertFalse(getattr(layer, "_hcu_slimquant_w4a8_hipc", False))

    def test_other_moe_calls_delegate_to_the_effective_original(self):
        normal = hooks.D.DeepEPNormalDispatchOutput(None, None, None, None, [])
        masked = hooks.D.DeepEPLLDispatchOutput(None, None, None, None, None, 0)
        for layer, dispatch in (
            (SimpleNamespace(_hcu_slimquant_w4a8_hipc=True), normal),
            (SimpleNamespace(), masked),
        ):
            with self.subTest(dispatch=dispatch), patch.object(hooks, "_is_hcu", True):
                original = Mock()
                actual = hooks.around_run_moe_core(original, layer, dispatch)
                original.assert_called_once_with(layer, dispatch)
                self.assertIs(actual, original.return_value)

    def test_contiguous_moe_scatters_and_gathers_weighted_outputs(self):
        import deepgemm

        x = torch.ones(2, 8, dtype=torch.int8)
        scales = torch.ones(2, 1)
        ids = torch.tensor([[0, 1], [1, -1]])
        weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        counts = [256, 256]
        dispatch = hooks.D.DeepEPNormalDispatchOutput(x, scales, ids, weights, counts)
        layer = SimpleNamespace(
            deprecate_flag=False,
            use_w4afp8=False,
            quant_config=object(),
            _hcu_slimquant_w4a8_contiguous_hipc=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            w13_weight=torch.empty(2, 8, 4, dtype=torch.int8),
            w13_weight_scale=torch.ones(2, 8, 1),
            w2_weight=torch.empty(2, 8, 2, dtype=torch.int8),
            w2_weight_scale=torch.ones(2, 8, 1),
        )
        segments = torch.tensor([0] * 256 + [1] * 256, dtype=torch.int32)
        indices = torch.tensor([[0, 256], [257, -1]], dtype=torch.int32)
        quantized = (torch.ones(512, 4, dtype=torch.int8), torch.ones(512, 1))

        def gemm(a, b, out, m_indices):
            self.assertIs(m_indices, segments)
            out[:256].fill_(2)
            out[256:].fill_(6)

        def gather(down, topk_ids, topk_weights, output_index, result):
            self.assertIs(topk_ids, ids)
            self.assertIs(topk_weights, weights)
            self.assertIs(output_index, indices)
            values = down[output_index.clamp_min(0)]
            result.copy_((values * topk_weights.unsqueeze(-1)).sum(dim=1))

        with (
            patch.object(hooks, "_is_hcu", True),
            patch.object(hooks, "get_offloader", return_value=SimpleNamespace(forbid_copy_engine_usage=True)),
            patch.object(hooks, "copy_list_to_gpu_no_ce", return_value=torch.tensor(counts, dtype=torch.int32)),
            patch.object(hooks, "_ep_scatter_with_optional_lightop", return_value=(segments, indices)) as scatter,
            patch.object(hooks, "_ep_gather_with_optional_lightop", side_effect=gather) as gather_op,
            patch.object(deepgemm, "m_grouped_w4a8_gemm_nt_contiguous_hipc", side_effect=gemm) as op,
            patch("lightop.activation.fuse_silu_mul_quant", return_value=quantized),
        ):
            combined = hooks.around_run_moe_core(DeepEPMoE.run_moe_core, layer, dispatch)

        self.assertIs(scatter.call_args.args[0], x)
        self.assertIs(scatter.call_args.args[1], scales)
        self.assertTrue(scatter.call_args.kwargs["counts_are_aligned"])
        self.assertEqual(op.call_count, 2)
        self.assertIs(op.call_args_list[1].args[0], quantized)
        gather_op.assert_called_once()
        self.assertIs(combined.topk_ids, ids)
        self.assertIs(combined.topk_weights, weights)
        torch.testing.assert_close(combined.hidden_states, torch.tensor([[5.] * 8, [6.] * 8], dtype=torch.bfloat16))

    def test_contiguous_dispatch_produces_int8_with_aligned_counts(self):
        from hcu_sglang.layers.moe import deepep_current as d
        from hcu_sglang.layers.moe import deepep_w4a8

        dispatcher = object.__new__(d.DeepEPDispatcher)
        dispatcher.deepep_mode = DeepEPMode.AUTO
        dispatcher.quant_config = {"existing": True}
        dispatcher._low_latency_dispatcher = Mock()
        normal = object.__new__(d._DeepEPDispatcherImplNormal)
        normal.num_experts = 16
        normal.async_finish = False
        normal.set_deepep_dispatcher_dtype = Mock()
        dispatcher._normal_dispatcher = normal
        layer = SimpleNamespace(
            dispatcher=dispatcher, moe_runner_config=SimpleNamespace(activation="silu")
        )

        x = torch.ones(1, 8, dtype=torch.bfloat16)
        ids = torch.tensor([[2, -1]], dtype=torch.int64)
        weights = torch.tensor([[1., 0.]])
        q = (x.to(torch.int8), torch.ones(1, 1))
        counts = [256, 0]
        buffer = Mock()
        buffer.get_dispatch_layout.return_value = (None, None, None, None, None)
        buffer.dispatch.return_value = (q, ids, weights, counts, object(), None)
        normal._get_buffer = Mock(return_value=buffer)

        with (
            patch.object(d, "_is_hcu", True),
            patch.object(d, "use_groupgemm", True),
            patch.object(d, "_hcu_deepgemm_is_selected", return_value=False),
            patch.object(d, "_deepep_precompile_tp_barrier"),
            patch.object(d, "get_global_expert_distribution_recorder"),
            patch.object(d.DeepEPConfig, "get_instance", return_value=SimpleNamespace(normal_dispatch_config=None)),
            patch.object(d, "per_token_quant_int8", return_value=q) as quantize,
        ):
            deepep_w4a8.configure_hipc(layer, masked=False, contiguous=True)
            packet = normal.dispatch_b(
                *normal.dispatch_a(x, SimpleNamespace(topk_ids=ids, topk_weights=weights))
            )

        self.assertTrue(layer._hcu_slimquant_w4a8_contiguous_hipc)
        self.assertFalse(layer._hcu_slimquant_w4a8_hipc)
        self.assertTrue(normal.quant_config["existing"])
        self.assertEqual(normal.quant_config["normal_dispatcher_output_dtype"], "int8")
        ll_config = dispatcher._low_latency_dispatcher.set_quant_config.call_args.args[0]
        self.assertNotIn("hcu_w4a8_int8_dispatch", ll_config)
        quantize.assert_called_once_with(x)
        self.assertEqual(buffer.dispatch.call_args.kwargs["expert_alignment"], 256)
        self.assertIs(packet.hidden_states, q[0])
        self.assertIs(packet.hidden_states_scale, q[1])
        self.assertIs(packet.num_recv_tokens_per_expert, counts)

    def test_contiguous_empty_dispatch_and_invalid_alignment(self):
        from hcu_sglang.layers.moe import deepep_w4a8

        dispatch = hooks.D.DeepEPNormalDispatchOutput(
            torch.empty(0, 8, dtype=torch.int8), None,
            torch.empty(0, 2, dtype=torch.int64), torch.empty(0, 2), [0, 0],
        )
        output = deepep_w4a8.run_contiguous_moe_core(object(), dispatch)
        self.assertEqual(output.hidden_states.shape, (0, 8))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "256-aligned"):
            deepep_w4a8.run_contiguous_moe_core(
                object(), dispatch._replace(num_recv_tokens_per_expert=[255, 1])
            )

class DeepEPContiguousWidthTests(unittest.TestCase):
    def load_layer(self, n1, n2):
        import deepgemm
        from hcu_sglang.layers.moe import deepep_w4a8
        from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as quant

        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(activation="silu", swiglu_limit=None)
        for name, tensor in (
            ("w13_weight", torch.ones(2, n1, 32, dtype=torch.int8)),
            ("w2_weight", torch.ones(2, n2, n1 // 4, dtype=torch.int8)),
            ("w13_weight_scale", torch.ones(2, n1, 1)),
            ("w2_weight_scale", torch.ones(2, n2, 1)),
        ):
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
        method = object.__new__(quant.SlimQuantW4A8Int8MarlinMoEMethod)
        method.use_deepep = True
        seen = []

        def pack(weight):
            if weight.shape[1] % 512:
                raise RuntimeError("W4A8 WASP mode 1006 requires N divisible by 512")
            seen.append(weight.shape[1])
            return weight

        with (
            patch.object(quant, "get_moe_a2a_backend", return_value=SimpleNamespace(is_megamoe=lambda: False)),
            patch.object(quant, "_is_hcu", True),
            patch.object(quant, "_use_lightop_w4a8_marlin_moe", False),
            patch.object(quant, "get_bool_env_var", side_effect=lambda name: name == "SGLANG_USE_W4A8_CONTIGUOUS_HIPC"),
            patch.object(deepgemm, "pack_w4a8_moe_hipc_weight", side_effect=pack),
            patch.object(deepep_w4a8, "configure_hipc"),
        ):
            method.process_weights_after_loading(layer)
        return layer, seen

    def test_model_width_is_padded_before_vendor_pack(self):
        layer, seen = self.load_layer(1280, 2560)
        self.assertEqual(seen, [1536, 2560])
        self.assertEqual(layer._hcu_w4a8_hipc_output_sizes, (1280, 2560))
        self.assertTrue(torch.all(layer.w13_weight[:, :1280] == 1))
        self.assertTrue(torch.all(layer.w13_weight[:, 1280:] == 0))
        self.assertTrue(torch.all(layer.w13_weight_scale[:, :1280] == 1))
        self.assertTrue(torch.all(layer.w13_weight_scale[:, 1280:] == 0))

    def test_padding_is_removed_before_silu_and_weighted_gather(self):
        import deepgemm
        from hcu_sglang.layers.moe import deepep_w4a8

        layer, seen = self.load_layer(1280, 64)
        self.assertEqual(seen, [1536, 512])
        x = torch.ones(2, 64, dtype=torch.int8)
        ids = torch.tensor([[0, 1], [1, -1]])
        weights = torch.tensor([[.25, .75], [1., 0.]])
        dispatch = hooks.D.DeepEPNormalDispatchOutput(x, torch.ones(2, 1), ids, weights, [256, 256])
        segments = torch.tensor([0] * 256 + [1] * 256, dtype=torch.int32)
        index = torch.tensor([[0, 256], [257, -1]], dtype=torch.int32)

        def gemm(a, b, out, m_indices):
            self.assertIs(m_indices, segments)
            out.fill_(float("nan"))
            width = 1280 if out.shape[1] == 1536 else 64
            out[:256, :width].fill_(2)
            out[256:, :width].fill_(6)

        def quantize(gateup):
            self.assertEqual(gateup.shape, (512, 1280))
            self.assertTrue(gateup.is_contiguous())
            self.assertTrue(torch.isfinite(gateup).all())
            return torch.ones(512, 640, dtype=torch.int8), torch.ones(512, 1)

        def gather(down, topk_ids, topk_weights, output_index, result):
            self.assertEqual(down.shape, (512, 64))
            self.assertTrue(down.is_contiguous())
            values = down[output_index.clamp_min(0)]
            result.copy_((values * topk_weights.unsqueeze(-1)).sum(dim=1))

        with (
            patch.object(hooks, "get_offloader", return_value=SimpleNamespace(forbid_copy_engine_usage=True)),
            patch.object(hooks, "copy_list_to_gpu_no_ce", return_value=torch.tensor([256, 256], dtype=torch.int32)),
            patch.object(hooks, "_ep_scatter_with_optional_lightop", return_value=(segments, index)),
            patch.object(hooks, "_ep_gather_with_optional_lightop", side_effect=gather),
            patch.object(deepgemm, "m_grouped_w4a8_gemm_nt_contiguous_hipc", side_effect=gemm),
            patch("lightop.activation.fuse_silu_mul_quant", side_effect=quantize),
        ):
            output = deepep_w4a8.run_contiguous_moe_core(layer, dispatch)
        torch.testing.assert_close(output.hidden_states, torch.tensor([[5.] * 64, [6.] * 64], dtype=torch.bfloat16))


    def test_aligned_weights_keep_their_widths(self):
        layer, seen = self.load_layer(1024, 512)
        self.assertEqual(seen, [1024, 512])
        self.assertEqual(layer._hcu_w4a8_hipc_output_sizes, (1024, 512))

    def test_auto_low_latency_removes_the_shared_weight_padding(self):
        import deepgemm
        from hcu_sglang.layers.moe import deepep_w4a8

        layer, _ = self.load_layer(1280, 64)
        x = torch.ones(2, 4, 64, dtype=torch.int8)
        dispatch = hooks.D.DeepEPLLDispatchOutput(x, torch.ones(2, 4, 1), torch.tensor([[0, -1]]), torch.tensor([[1., 0.]]), torch.tensor([4, 0], dtype=torch.int32), 4)

        def gemm(a, b, out, mask, expected):
            out.fill_(float("nan"))
            out[..., :1280 if out.shape[-1] == 1536 else 64].fill_(2)

        def quantize(gateup, mask):
            self.assertEqual(gateup.shape, (2, 4, 1280))
            self.assertTrue(gateup.is_contiguous())
            self.assertTrue(torch.isfinite(gateup).all())
            return torch.ones(2, 4, 640, dtype=torch.int8), torch.ones(2, 4, 1)

        with (
            patch.object(deepgemm, "m_grouped_w4a8_gemm_nt_masked_hipc", side_effect=gemm),
            patch("lightop.activation.fuse_silu_mul_quant_ep", side_effect=quantize),
        ):
            output = deepep_w4a8.run_masked_moe_core(layer, dispatch)
        self.assertEqual(output.hidden_states.shape, (2, 4, 64))
        self.assertTrue(output.hidden_states.is_contiguous())
        torch.testing.assert_close(output.hidden_states, torch.full((2, 4, 64), 2, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
