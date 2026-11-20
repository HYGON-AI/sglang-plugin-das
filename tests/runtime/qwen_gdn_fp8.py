from contextlib import ExitStack, nullcontext
import sys
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from hcu_sglang.environ import register_envs

register_envs()

from compressed_tensors.quantization import QuantizationArgs
from sglang.srt.layers import linear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.model_executor import forward_context
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3_5 as native
from sglang.srt.plugins.hook_registry import HookRegistry

from hcu_sglang.layers.quantization import compressed_fp8, fp8_linear
from hcu_sglang.models.qwen import gdn_fp8, qwen3_5


class QwenGdnFp8Tests(TestCase):
    def setUp(self):
        self.ctx = self.enterContext(ExitStack())
        self.ctx.enter_context(patch.object(compressed_fp8, "_is_hcu", True))
        self.ctx.enter_context(patch.object(compressed_fp8, "_use_lightop_channel_fp8", False))
        self.ctx.enter_context(patch.object(compressed_fp8.Q, "_use_aiter", False))
        self.tokens, self.heads, self.width = 2, 24, 128
        self.x = torch.ones(self.tokens, self.heads, self.width, dtype=torch.bfloat16)
        full_gate = torch.zeros(self.tokens, self.heads, 2 * self.width, dtype=self.x.dtype)
        self.z = full_gate[..., self.width:]
        self.q = torch.ones(self.tokens, self.heads * self.width).to(torch.float8_e4m3fn)
        self.scale = torch.tensor([[0.125], [0.25]])

        self.proj = linear.RowParallelLinear.__new__(linear.RowParallelLinear)
        torch.nn.Module.__init__(self.proj)
        self.proj.scheme = compressed_fp8.Q.CompressedTensorsW8A8Fp8(
            QuantizationArgs(num_bits=8, type="float", strategy="channel"), False
        )
        self.proj.weight = torch.ones(self.heads * self.width, 3).to(torch.float8_e4m3fn)
        self.proj.weight_scale = torch.full((3, 1), 0.25)
        self.proj.tp_group = object()
        self.proj.parallel_group = "tp"
        self.proj.input_is_parallel = True
        self.proj.use_dp_attention_reduce = False
        self.proj.use_decode_attn_tp = False
        self.proj.reduce_results = True
        self.proj.skip_bias_add = False
        self.proj.bias = None
        self.proj.quant_method = CompressedTensorsLinearMethod(None)
        self.norm = Mock(
            weight=torch.ones(128, dtype=torch.bfloat16), eps=1e-6,
            norm_before_gate=True, group_size=None, activation="swish", bias=None,
        )
        self.layer = SimpleNamespace(
            norm=self.norm, out_proj=self.proj, head_v_dim=128, head_k_dim=128,
            num_v_heads=48, num_k_heads=16, attn_tp_size=2,
            _forward_input_proj=Mock(return_value=(self.x, self.x)),
        )
        self.launch_input = SimpleNamespace(
            numel=lambda: 1, is_cuda=True, dtype=torch.bfloat16,
            device=SimpleNamespace(index=0),
        )

    def test_guards_dynamic_channel_fp8_and_actual_arch(self):
        self.enterContext(patch.object(
            type(self.proj.scheme), "supports_fp8_prequantized_input",
            compressed_fp8._PREQUANTIZED_CAPABILITY, create=True,
        ))
        with patch.object(gdn_fp8, "_is_gfx938", return_value=True):
            self.assertTrue(gdn_fp8._supports_fusion(self.layer, self.launch_input))
            self.proj.scheme.is_static_input_scheme = True
            self.assertFalse(gdn_fp8._supports_fusion(self.layer, self.launch_input))
            self.proj.scheme.is_static_input_scheme = False
            with patch.object(compressed_fp8.Q, "_use_aiter", True):
                self.assertFalse(gdn_fp8._supports_fusion(self.layer, self.launch_input))
            self.proj.weight = self.proj.weight.to(torch.float8_e4m3fnuz)
            self.assertFalse(gdn_fp8._supports_fusion(self.layer, self.launch_input))
        with patch.object(gdn_fp8, "_is_gfx938", return_value=False):
            self.proj.weight = self.proj.weight.to(torch.float8_e4m3fn)
            self.assertFalse(gdn_fp8._supports_fusion(self.layer, self.launch_input))

    def test_opt_in_registers_gdn_forward(self):
        target = "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward"
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(qwen3_5, "get_bool_env_var", return_value=enabled),
                patch.object(HookRegistry, "register") as register,
            ):
                qwen3_5.register()
            calls = [call for call in register.call_args_list if call.args[0] == target]
            self.assertEqual(len(calls), int(enabled))
            if enabled:
                self.assertIs(calls[0].args[1], gdn_fp8.around_forward)

    def test_provider_selection_with_current_and_new_lightop(self):
        from hcu_sglang.layers.attention.fla import gated_norm_fp8

        self.addCleanup(gdn_fp8._kernel.cache_clear)
        gdn_fp8._kernel.cache_clear()
        with patch.dict(sys.modules, {"lightop.norm": SimpleNamespace()}):
            self.assertIs(
                gdn_fp8._kernel(), gated_norm_fp8.gated_rms_norm_per_token_fp8_quant
            )
        gdn_fp8._kernel.cache_clear()
        provider = Mock()
        module = SimpleNamespace(gated_rms_norm_per_token_fp8_quant=provider)
        with patch.dict(sys.modules, {"lightop.norm": module}):
            self.assertIs(gdn_fp8._kernel(), provider)

    def test_disabled_or_sequence_parallel_uses_effective_original(self):
        original = Mock(return_value=self.x)
        with patch.object(gdn_fp8, "enabled", False):
            self.assertIs(gdn_fp8.around_forward(original, self.layer, self.x, None), self.x)
        with (
            patch.object(gdn_fp8, "enabled", True),
            patch.object(gdn_fp8, "_supports_fusion", return_value=True),
            patch.object(gdn_fp8, "get_forward", return_value=SimpleNamespace(sp_active=True)),
        ):
            self.assertIs(gdn_fp8.around_forward(original, self.layer, self.x, None), self.x)
        self.assertEqual(original.call_count, 2)
        self.layer._forward_input_proj.assert_not_called()

    def test_gdn_passes_one_fp8_pair_to_gemm_and_tp2_reduce(self):
        flags = SimpleNamespace(sp_active=False)
        cases = (
            (ForwardMode.EXTEND, False),
            (ForwardMode.EXTEND, True),
            (ForwardMode.DECODE, False),
            (ForwardMode.TARGET_VERIFY, False),
        )
        for mode, strided in cases:
            with self.subTest(mode=mode, strided=strided), ExitStack() as stack:
                prefill = mode == ForwardMode.EXTEND
                self.layer.num_k_heads = 12 if strided else 16
                z = self.z if strided else self.z.contiguous()
                self.layer.fix_query_key_value_ordering = Mock(
                    return_value=(self.x, self.x, self.x, z, self.x, self.x)
                )
                self.layer.attn = Mock(return_value=self.x if prefill else (self.x, z))
                collective = Mock(side_effect=lambda x: x + 1)
                gemm = Mock(side_effect=lambda _a, _b, out: out.fill_(2))
                kernel = Mock(return_value=(self.q, self.scale))
                original = Mock(side_effect=AssertionError("unexpected fallback"))
                ctx = SimpleNamespace(tp_group=SimpleNamespace(all_reduce=collective))
                replacements = (
                    (gdn_fp8, "enabled", True),
                    (gdn_fp8, "_supports_fusion", Mock(return_value=True)),
                    (gdn_fp8, "get_forward", Mock(return_value=flags)),
                    (gdn_fp8, "_kernel", Mock(return_value=kernel)),
                    (native, "_is_cuda", True),
                    (native, "_is_cpu", False),
                    (native, "_GDN_FUSED_QKVZBA_RATIOS", (1, 2, 4, 8)),
                    (native, "_gdn_decode_fused_proj_conv", True),
                    (native, "qwen3_5_gdn_prefill_projection_views", Mock(return_value=(self.x, z, self.x, self.x))),
                    (forward_context, "get_attn_backend", Mock(return_value=SimpleNamespace(supports_packed_verify=True))),
                    (linear, "get_group_rank_size", Mock(return_value=(1, 2))),
                    (linear, "get_forward", Mock(return_value=flags)),
                    (linear, "get_parallel", Mock(return_value=ctx)),
                    (linear, "is_allocation_symmetric", Mock(return_value=False)),
                    (linear, "use_symmetric_memory", Mock(return_value=nullcontext())),
                    (linear, "should_skip_mlp_all_reduce", Mock(return_value=False)),
                    (compressed_fp8.Q, "apply_fp8_linear", fp8_linear.apply_fp8_linear),
                    (compressed_fp8.Q.CompressedTensorsW8A8Fp8, "apply_weights", compressed_fp8.apply_weights),
                    (fp8_linear, "_is_hcu", True),
                    (fp8_linear.deepgemm, "fp8_gemm", gemm),
                    (fp8_linear.N, "scaled_fp8_quant", Mock(side_effect=AssertionError("requantization"))),
                )
                for obj, name, value in replacements:
                    stack.enter_context(patch.object(obj, name, value))
                out = gdn_fp8.around_forward(
                    original, self.layer, self.x, SimpleNamespace(forward_mode=mode)
                )
                self.layer.attn.assert_called_once()
                self.norm.assert_not_called()
                kernel.assert_called_once()
                gate = kernel.call_args.args[1]
                self.assertEqual(gate.shape, z.shape if strided else (self.tokens * self.heads, self.width))
                self.assertEqual(gate.stride(-1), 1)
                if strided:
                    self.assertIs(gate, z)
                gemm.assert_called_once()
                self.assertEqual(gemm.call_args.args[0][0].data_ptr(), self.q.data_ptr())
                self.assertIs(gemm.call_args.args[0][1], self.scale)
                collective.assert_called_once()
                torch.testing.assert_close(out, torch.full((self.tokens, 3), 3, dtype=torch.bfloat16))


if __name__ == "__main__":
    main()
