# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class TestMiniMaxDSparkInt8Kv(unittest.TestCase):
    def setUp(self):
        from compressed_tensors.quantization import QuantizationStrategy
        from hcu_sglang.models import minimax_m31_dspark as plugin
        class W8A8Scheme:
            strategy = QuantizationStrategy.CHANNEL
            is_static_input_scheme = False
            input_symmetric = True
            w8a8_strategy = 3

        source_module = (
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8"
        )
        scheme_module = SimpleNamespace(CompressedTensorsW8A8Int8=W8A8Scheme)
        imports = patch.dict(sys.modules, {source_module: scheme_module})
        imports.start()
        self.addCleanup(imports.stop)

        self.plugin = plugin
        self.original = Mock()
        self.hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        scheme = W8A8Scheme()

        def apply_weights(layer, hidden, bias):
            weight = layer.weight.T if scheme.w8a8_strategy is None else layer.weight
            output = hidden @ (weight.float() * layer.weight_scale).T
            return output if bias is None else output + bias

        scheme.apply_weights = Mock(side_effect=apply_weights)
        self.projection = SimpleNamespace(
            weight=torch.arange(32, dtype=torch.int8).reshape(8, 4),
            weight_scale=torch.arange(1, 9, dtype=torch.float32).reshape(8, 1),
            bias=torch.arange(8, dtype=torch.float32),
            skip_bias_add=False,
            scheme=scheme,
        )
        self.attention = SimpleNamespace(
            qkv_proj=self.projection, q_size=4, kv_size=2, hidden_size=4
        )

    def test_gemm_consumes_only_kv_rows_and_matching_scale_and_bias(self):
        with patch.object(self.plugin, "is_hcu", return_value=True):
            for bias in (self.projection.bias, None):
                self.projection.bias = bias
                k, v = self.plugin._draft_int8_kv_only(
                    self.original, self.attention, self.hidden
                )
                weight = self.projection.weight.float() * self.projection.weight_scale
                full = self.hidden @ weight.T
                if bias is not None:
                    full = full + bias
                torch.testing.assert_close(k, full[:, 4:6])
                torch.testing.assert_close(v, full[:, 6:8])
                sliced = self.projection.scheme.apply_weights.call_args.args[0]
                self.assertEqual(sliced.weight.shape, (4, 4))
                self.assertEqual(sliced.weight.data_ptr(), self.projection.weight[4:].data_ptr())
        self.original.assert_not_called()

    def test_community_layout_skips_q_without_changing_int8_backend(self):
        row_weight = self.projection.weight
        self.projection.weight = row_weight.T
        self.projection.scheme.w8a8_strategy = None
        with patch.object(self.plugin, "is_hcu", return_value=True):
            k, v = self.plugin._draft_int8_kv_only(
                self.original, self.attention, self.hidden
            )
        full = self.hidden @ (row_weight.float() * self.projection.weight_scale).T
        full = full + self.projection.bias
        torch.testing.assert_close(torch.cat((k, v), dim=-1), full[:, 4:])
        sliced = self.projection.scheme.apply_weights.call_args.args[0]
        self.assertEqual(sliced.weight.stride(), self.projection.weight.stride())
        self.assertEqual(sliced.weight.data_ptr(), self.projection.weight[:, 4:].data_ptr())
        self.original.assert_not_called()

    def test_unsupported_scheme_and_opt_out_keep_original(self):
        hook = self.plugin._draft_int8_kv_only
        with patch.object(self.plugin, "is_hcu", return_value=False):
            self.assertIs(hook(self.original, self.attention, self.hidden), self.original.return_value)
        with patch.object(self.plugin, "is_hcu", return_value=True):
            with patch.object(self.plugin, "_INT8_KV_ONLY", False):
                self.assertIs(hook(self.original, self.attention, self.hidden), self.original.return_value)
            self.projection.scheme.w8a8_strategy = 1
            self.assertIs(hook(self.original, self.attention, self.hidden), self.original.return_value)
            self.projection.scheme.apply_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
