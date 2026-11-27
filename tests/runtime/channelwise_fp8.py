"""CPU loader/dispatch guards; device arithmetic is covered by the HCU probe."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.layers.quantization import channelwise_fp8 as runtime
from hcu_sglang.environ import register_envs

register_envs()


class TestChannelwiseRuntime(unittest.TestCase):
    def test_real_v41_config_normalization_is_supported(self):
        from hcu_sglang.configs import (
            _per_channel_fp8_as_compressed_tensors,
        )
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        config = CompressedTensorsConfig.from_config(
            _per_channel_fp8_as_compressed_tensors(
                {
                    "quant_method": "fp8",
                    "is_per_channel": True,
                    "activation_scheme": "dynamic",
                }
            )
        )
        with (
            patch.dict(os.environ, {"SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME": "1"}),
            patch("hcu_sglang.utils.detection.is_hcu", return_value=True),
        ):
            self.assertTrue(runtime.enabled(config))

    def test_disabled_is_compatible_with_other_quantization(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME": "0"}):
            self.assertFalse(runtime.enabled(None))
            self.assertFalse(runtime.enabled(object()))

    def test_load_requires_both_fp8_weight_and_channel_scale(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        runtime.install(layer)
        weight = torch.arange(32).reshape(4, 8).to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        with self.assertRaises(ValueError):
            layer.quant_method.process_weights_after_loading(layer)
        layer.weight.weight_loader(layer.weight, weight)
        with self.assertRaises(ValueError):
            layer.quant_method.process_weights_after_loading(layer)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
        layer.quant_method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight.float(), weight.float())
        torch.testing.assert_close(layer.weight_scale_inv, scale)

    def test_rejects_non_fp8_or_block_scales(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        runtime.install(layer)
        with self.assertRaises(ValueError):
            layer.weight.weight_loader(layer.weight, torch.ones(4, 8))
        with self.assertRaises(ValueError):
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch.ones(1))

    def test_weight_and_scale_use_same_tp_slice_and_padding(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        layer.weight.output_dim = 0

        def loader(param, value):
            self.assertEqual(param.output_dim, 0)
            # Simulate a final vocabulary partition containing three real rows.
            param.data[:3].copy_(value[4:7])
            param.data[3:].zero_()

        layer.weight.weight_loader = loader
        runtime.install(layer)
        weight = torch.arange(56).reshape(7, 8).to(torch.float8_e4m3fn)
        scale = torch.arange(1, 8).float().reshape(7, 1)
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
        torch.testing.assert_close(layer.weight[:3].float(), weight[4:7].float())
        torch.testing.assert_close(layer.weight_scale_inv[:3], scale[4:7])
        self.assertEqual(layer.weight_scale_inv[3].item(), 0)

    def test_grouped_keeps_group_scale_alignment(self):
        x = torch.randn(5, 2, 8)
        layer = SimpleNamespace(
            weight=torch.randn(6, 8), weight_scale_inv=torch.arange(6).reshape(6, 1)
        )

        def project(a, w, s):
            self.assertEqual(a.shape, (2, 5, 8))
            torch.testing.assert_close(s[:, :, 0], torch.arange(6).reshape(2, 3))
            return torch.matmul(a, w.transpose(-1, -2))

        with patch.object(runtime, "linear", side_effect=project):
            actual = runtime.grouped(x, layer, 2, 3)
        expected = torch.einsum("tgd,grd->tgr", x, layer.weight.reshape(2, 3, 8))
        torch.testing.assert_close(actual, expected)

    def test_fp32_batches_dispatch_separately(self):
        x = torch.randn(2, 3, 8)
        w = torch.randn(2, 4, 8).to(torch.float8_e4m3fn)
        scale = torch.ones(2, 4, 1)
        calls = []

        def quant(a):
            return a.to(torch.float8_e4m3fn), torch.ones(*a.shape[:-1], 1)

        def gemm(a, b, out):
            # Reproduce the backend restriction instead of silently accepting
            # the bad batched-FP32 dispatch which this regression guards.
            self.assertEqual(b[0].ndim, 2)
            calls.append(b[0].shape)
            out.copy_((a[0].float() * a[1]) @ (b[0].float() * b[1]).T)

        with patch.dict(
            sys.modules,
            {
                "deepgemm": SimpleNamespace(fp8_gemm=gemm),
                "lightop.quant": SimpleNamespace(per_token_quant_fp8=quant),
            },
        ):
            y = runtime.linear(x, w, scale)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            y, x.to(torch.float8_e4m3fn).float() @ w.float().transpose(-1, -2)
        )


if __name__ == "__main__":
    unittest.main()
