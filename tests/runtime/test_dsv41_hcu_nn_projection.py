import os
import unittest
from unittest.mock import patch

import torch

from hcu_sglang.environ import register_envs


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "HCU required")
class TestNNProjection(unittest.TestCase):
    def test_postload_and_reload(self):
        register_envs()
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import bf16_fp32 as op
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer = torch.nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.weight.requires_grad_(False)
        layer.quant_method = UnquantizedLinearMethod()
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"SGLANG_HCU_ROCBLAS_NN": "1", "WEIGHT_CACHE": "daemon"}):
            op.prepare_hcu_nn_weights(layer)
            self.assertIsNone(op._nn_operand(layer.weight))
        with patch.dict(os.environ, {"SGLANG_HCU_ROCBLAS_NN": "1", "WEIGHT_CACHE": "client"}):
            op.prepare_hcu_nn_weights(layer)
            self.assertTrue(op._nn_operand(layer.weight).is_contiguous())
            out = op._linear_bf16_fp32_cublas(x, layer.weight)
            torch.testing.assert_close(out, torch.mm(x, layer.weight.T).float(), rtol=.008, atol=.002)
            layer.weight.zero_()
            self.assertIsNone(op._nn_operand(layer.weight))
            op.prepare_hcu_nn_weights(layer)
            self.assertEqual(op._linear_bf16_fp32_cublas(x, layer.weight).count_nonzero().item(), 0)
