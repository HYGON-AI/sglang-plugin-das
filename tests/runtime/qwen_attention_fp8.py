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
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3_5 as native
from sglang.srt.plugins.hook_registry import HookRegistry

from hcu_sglang.layers.quantization import compressed_fp8, fp8_linear
from hcu_sglang.models.qwen import attention_fp8, qwen3_5


class QwenAttentionFp8Tests(TestCase):
    def setUp(self):
        self.x = torch.ones((2, 3072), dtype=torch.bfloat16)
        self.gate = torch.zeros((2, 12, 512), dtype=torch.bfloat16)[..., 256:]
        self.q = self.x.to(torch.float8_e4m3fn)
        self.scale = torch.tensor([[0.125], [0.25]])
        self.proj = linear.RowParallelLinear.__new__(linear.RowParallelLinear)
        torch.nn.Module.__init__(self.proj)
        self.proj.scheme = compressed_fp8.Q.CompressedTensorsW8A8Fp8(
            QuantizationArgs(num_bits=8, type="float", strategy="channel"), False
        )
        self.proj.weight = torch.ones((3072, 3)).to(torch.float8_e4m3fn)
        self.proj.weight_scale = torch.full((3, 1), 0.25)
        self.proj.tp_group = object()
        self.proj.parallel_group = "tp"
        self.proj.input_is_parallel = True
        self.proj.input_size_per_partition = 3072
        self.proj.use_dp_attention_reduce = False
        self.proj.use_decode_attn_tp = False
        self.proj.reduce_results = True
        self.proj.skip_bias_add = False
        self.proj.bias = None
        self.proj.quant_method = CompressedTensorsLinearMethod(None)
        self.layer = SimpleNamespace(o_proj=self.proj, attn_output_gate=True)
        self.layer.attn = Mock(return_value=self.x)

    def test_registers_once_and_keeps_default_hook(self):
        for enabled, expected in (
            (False, qwen3_5.replacement_self_attention),
            (True, attention_fp8.around_self_attention),
        ):
            with (
                self.subTest(enabled=enabled),
                patch.object(qwen3_5, "get_bool_env_var", return_value=enabled),
                patch.object(HookRegistry, "register") as register,
            ):
                qwen3_5.register()
            calls = [call for call in register.call_args_list if call.args[0] == qwen3_5._TARGET]
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0].args[1], expected)

    def test_scheme_arch_and_local_partition_gate(self):
        x = SimpleNamespace(
            numel=lambda: 1, is_cuda=True, dtype=torch.bfloat16,
            device=SimpleNamespace(index=0),
        )
        with (
            patch.object(compressed_fp8, "_is_hcu", True),
            patch.object(compressed_fp8.Q, "_use_aiter", False),
            patch.object(type(self.proj.scheme), "supports_fp8_prequantized_input",
                         compressed_fp8._PREQUANTIZED_CAPABILITY, create=True),
            patch.object(attention_fp8, "_is_gfx938", return_value=True) as arch,
        ):
            self.assertTrue(attention_fp8._supports_fusion(self.layer, x))
            arch.assert_called_once_with(0)
            arch.return_value = False
            self.assertFalse(attention_fp8._supports_fusion(self.layer, x))
            arch.return_value = True
            self.proj.scheme.is_static_input_scheme = True
            self.assertFalse(attention_fp8._supports_fusion(self.layer, x))
            self.proj.scheme.is_static_input_scheme = False
            self.proj.input_is_parallel = False
            self.assertFalse(attention_fp8._supports_fusion(self.layer, x))
            self.proj.input_is_parallel = True
            self.proj.weight = self.proj.weight.to(torch.float8_e4m3fnuz)
            self.assertFalse(attention_fp8._supports_fusion(self.layer, x))

    def test_provider_and_bundled_fallback(self):
        from hcu_sglang.layers.attention import sigmoid_mul_fp8

        self.addCleanup(attention_fp8._kernel.cache_clear)
        attention_fp8._kernel.cache_clear()
        with patch.dict(sys.modules, {"lightop.activation": SimpleNamespace()}):
            self.assertIs(attention_fp8._kernel(), sigmoid_mul_fp8.fuse_sigmoid_mul_fp8_quant)
        attention_fp8._kernel.cache_clear()
        provider = Mock()
        with patch.dict(sys.modules, {"lightop.activation": SimpleNamespace(fuse_sigmoid_mul_fp8_quant=provider)}):
            self.assertIs(attention_fp8._kernel(), provider)

    def test_uses_effective_original_when_sequence_parallel(self):
        original = Mock(return_value=self.x)
        with (
            patch.object(attention_fp8, "_supports_fusion", return_value=True),
            patch.object(attention_fp8, "get_forward", return_value=SimpleNamespace(sp_active=True)),
        ):
            result = attention_fp8.around_self_attention(original, self.layer, None, self.x, None)
        self.assertIs(result, self.x)
        original.assert_called_once_with(self.layer, None, self.x, None)
        self.layer.attn.assert_not_called()

    def test_preserves_fallback_after_attention_without_repeating_attention(self):
        gate = self.gate.to(torch.float32)
        self.layer._prepare_qkv_gate = Mock(return_value=(self.x, self.x, self.x, gate))
        self.layer.o_proj = Mock(return_value=(self.x, None), input_size_per_partition=3072)
        original = Mock()
        with (
            patch.object(attention_fp8, "_supports_fusion", return_value=True),
            patch.object(attention_fp8, "get_forward", return_value=SimpleNamespace(sp_active=False)),
            patch.object(native, "_fp8_tuple_input", return_value=False),
            patch.object(native, "fused_sigmoid_mul", return_value=self.x) as split,
            patch.object(attention_fp8, "_kernel") as provider,
        ):
            result = attention_fp8.around_self_attention(original, self.layer, None, self.x, None)
        self.assertIs(result, self.x)
        split.assert_called_once_with(self.x, gate, inplace=True)
        self.layer.attn.assert_called_once()
        original.assert_not_called()
        provider.assert_not_called()

    def test_fp8_pair_reaches_real_row_and_compressed_fp8_consumers(self):
        flags = SimpleNamespace(sp_active=False)
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode), ExitStack() as stack:
                gate = self.gate if mode == ForwardMode.EXTEND else self.gate.contiguous().view_as(self.x)
                self.layer._prepare_qkv_gate = Mock(return_value=(self.x, self.x, self.x, gate))
                self.layer.attn.reset_mock()
                kernel = Mock(return_value=(self.q, self.scale))
                gemm = Mock(side_effect=lambda _q, _x, _w, _s, result: result.fill_(2))
                collective = Mock(side_effect=lambda x: x + 1)
                ctx = SimpleNamespace(tp_group=SimpleNamespace(all_reduce=collective))
                replacements = (
                    (attention_fp8, "_supports_fusion", Mock(return_value=True)),
                    (attention_fp8, "get_forward", Mock(return_value=flags)),
                    (attention_fp8, "_kernel", Mock(return_value=kernel)),
                    (native, "_fp8_tuple_input", Mock(return_value=False)),
                    (native, "fused_sigmoid_mul", Mock(side_effect=AssertionError("separate activation"))),
                    (linear, "get_group_rank_size", Mock(return_value=(1, 2))),
                    (linear, "get_forward", Mock(return_value=flags)),
                    (linear, "get_parallel", Mock(return_value=ctx)),
                    (linear, "is_allocation_symmetric", Mock(return_value=False)),
                    (linear, "use_symmetric_memory", Mock(return_value=nullcontext())),
                    (linear, "should_skip_mlp_all_reduce", Mock(return_value=False)),
                    (compressed_fp8, "_is_hcu", True),
                    (compressed_fp8, "_use_lightop_channel_fp8", False),
                    (compressed_fp8.Q, "apply_fp8_linear", fp8_linear.apply_fp8_linear),
                    (compressed_fp8.Q.CompressedTensorsW8A8Fp8, "apply_weights", compressed_fp8.apply_weights),
                    (fp8_linear, "_is_hcu", True),
                    (fp8_linear, "fp8_gemm", gemm),
                    (fp8_linear.N, "scaled_fp8_quant", Mock(side_effect=AssertionError("requantization"))),
                )
                for obj, name, value in replacements:
                    stack.enter_context(patch.object(obj, name, value))
                original = Mock(side_effect=AssertionError("original"))
                out = attention_fp8.around_self_attention(
                    original, self.layer, None, self.x, SimpleNamespace(forward_mode=mode)
                )
                self.layer.attn.assert_called_once()
                kernel.assert_called_once_with(self.x, gate)
                gemm.assert_called_once()
                self.assertEqual(gemm.call_args.args[0].data_ptr(), self.q.data_ptr())
                self.assertIs(gemm.call_args.args[1], self.scale)
                collective.assert_called_once()
                torch.testing.assert_close(out, torch.full((2, 3), 3, dtype=torch.bfloat16))


if __name__ == "__main__":
    main()
