import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hcu_sglang.layers import fused_rms_quant as fused


class TestDsv41Qnorm(unittest.TestCase):
    def test_cpu_and_empty_fall_back_without_loading_lightop(self):
        norm = SimpleNamespace(weight=torch.ones(1280, dtype=torch.bfloat16), variance_epsilon=1e-20)
        with patch.object(fused, "_fp8_qnorm_provider", side_effect=AssertionError):
            for rows in (0, 2):
                q = torch.zeros((rows, 1280), dtype=torch.bfloat16)
                self.assertIsNone(fused.dsv41_qnorm_fp8_operand(q, norm, True))

    def test_hcu_q_norm_bf16_and_serialized_fp8(self):
        from hcu_sglang.utils.detection import is_hcu_native_fp8_supported

        if not is_hcu_native_fp8_supported():
            self.skipTest("requires gfx938 and LightOp")
        from lightop import rmsnorm

        self.assertGreaterEqual(rmsnorm.RMS_NORM_FP8_QUANT_VERSION, 2)
        device = torch.device("cuda")
        width = 1280
        rng = torch.Generator(device=device).manual_seed(108)
        def make(shape):
            return torch.randn(shape, device=device, generator=rng, dtype=torch.bfloat16)

        for rows in (1, 19, 165):
            host = make((rows, width + 512))
            if rows > 1:
                host[0].zero_()
            norm = SimpleNamespace(weight=make((width,)), variance_epsilon=1e-20)
            source = host[:, :width].clone()
            tail = host[:, width:].clone()
            def run(q, use_strided):
                normalized, (codes, scales, dtype) = fused.dsv41_qnorm_fp8_operand(q, norm, use_strided)
                self.assertEqual(dtype, torch.bfloat16)
                self.assertEqual(codes.dtype, torch.float8_e4m3fn)
                self.assertEqual(scales.dtype, torch.float32)
                self.assertEqual(scales.shape, (rows, 1))
                return normalized, codes.view(torch.uint8), scales
            copied = run(source.clone(), False)
            result = run(host[:, :width], True)
            for actual, expected in zip(result, copied):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(host[:, width:], tail, rtol=0, atol=0)
            if rows > 1:
                torch.testing.assert_close(host[:, :width], source, rtol=0, atol=0)
            norm_out, codes, scales = result
            x = source.float()
            expected = (x * torch.rsqrt((x*x).mean(-1, keepdim=True)+norm.variance_epsilon)
                        * norm.weight.float()).bfloat16()
            torch.testing.assert_close(norm_out, expected, rtol=0.009, atol=1e-5)
            scale_ref = (norm_out.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1/(448*512))
            torch.testing.assert_close(scales, scale_ref, rtol=2e-6, atol=0)
            source_f32 = norm_out.float()
            reciprocal_values = source_f32 * scales.reciprocal()
            divide_values = source_f32 / scales
            multiply_codes = reciprocal_values.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
            divide_codes = divide_values.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
            difference = (codes != multiply_codes) | (codes != divide_codes)
            details = {
                "rows": rows, "numel": codes.numel(),
                "native_vs_reciprocal": (codes != multiply_codes).sum().item(),
                "native_vs_division": (codes != divide_codes).sum().item(),
                "reciprocal_vs_division": (multiply_codes != divide_codes).sum().item(),
            }
            print(json.dumps(details), flush=True)
            if directory := os.environ.get("HCU_QNORM_TEST_DIAGNOSTICS"):
                i, j = difference.nonzero(as_tuple=True)
                torch.save({
                    "counts": details, "row": i.cpu(), "column": j.cpu(),
                    "normalized": norm_out[i,j].cpu(), "scale": scales[i,0].cpu(),
                    "reciprocal_values": reciprocal_values[i,j].cpu(),
                    "division_values": divide_values[i,j].cpu(),
                    "native_codes": codes[i,j].cpu(),
                    "reciprocal_codes": multiply_codes[i,j].cpu(),
                    "division_codes": divide_codes[i,j].cpu(),
                }, Path(directory) / f"rounding-{rows}.pt")
            torch.testing.assert_close(codes, multiply_codes, rtol=0, atol=0)
        invalid = make((4,1280)).T
        error_norm = SimpleNamespace(weight=make((4,)), variance_epsilon=1e-20)
        self.assertIsNone(fused.dsv41_qnorm_fp8_operand(invalid, error_norm, True))
        self.assertIsNone(fused.dsv41_qnorm_fp8_operand(source[:0], norm, True))


    def test_current_qnorm_to_compressed_linear(self):
        from hcu_sglang.utils.detection import is_hcu_native_fp8_supported

        if not is_hcu_native_fp8_supported():
            self.skipTest("requires gfx938")
        from compressed_tensors.quantization import QuantizationArgs
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from hcu_sglang.environ import register_envs
        register_envs()
        from hcu_sglang.layers.quantization import w8a8_fp8
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset()
        w8a8_fp8.register_hooks()
        HookRegistry.apply_hooks()
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
        from hcu_sglang.models.deepseek.current_v4.attention import MQALayer

        rows, width, outputs = 19, 1280, 256
        rng = torch.Generator(device="cuda").manual_seed(108)
        weights = (torch.randn(outputs, width, device="cuda", generator=rng)*0.25).to(torch.float8_e4m3fn)
        weight_scales = torch.linspace(0.125,0.75,outputs,device="cuda").reshape(-1,1)
        layer = torch.nn.Module()
        layer.scheme = CompressedTensorsW8A8Fp8(
            QuantizationArgs(num_bits=8, type="float", strategy="channel"), False
        )
        method = CompressedTensorsLinearMethod(None)
        with torch.device("cuda"):
            method.create_weights(layer,width,[outputs],width,outputs,torch.bfloat16,weight_loader=lambda *_: None)
        layer.weight.data.copy_(weights)
        layer.weight_scale.data.copy_(weight_scales)
        method.process_weights_after_loading(layer)
        self.assertTrue(layer.scheme.supports_fp8_prequantized_input)
        attn = SimpleNamespace(
            _hcu_qnorm_fp8=True, _hcu_qnorm_strided=True,
            q_norm=SimpleNamespace(weight=torch.ones(width,device="cuda",dtype=torch.bfloat16), variance_epsilon=1e-20),
        )
        packed = torch.randn(rows,width+512,device="cuda",dtype=torch.bfloat16,generator=rng)
        indexer_bf16, operand = MQALayer._normalize_q_lora(attn, packed[:, :width])
        self.assertEqual(indexer_bf16.dtype,torch.bfloat16)
        q, s, dtype = operand
        output = method.apply(layer, operand)
        expected = ((q.float()*s) @ (weights.float()*weight_scales).T).to(dtype)
        torch.testing.assert_close(output, expected, rtol=.025, atol=.025)

        def project():
            indexer, (values, scales, dtype) = MQALayer._normalize_q_lora(attn, packed[:, :width])
            return indexer, values, scales, method.apply(layer, (values, scales, dtype))

        for _ in range(3):
            project()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = project()
        initial_ptrs = tuple(t.data_ptr() for t in captured)
        last_scale = None
        for columns in (32, 256):
            packed.normal_(generator=rng)
            packed[:, :columns].mul_(8)
            eager = project()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(tuple(t.data_ptr() for t in captured), initial_ptrs)
            torch.testing.assert_close(captured[0], eager[0], rtol=0, atol=0)
            torch.testing.assert_close(captured[1].view(torch.uint8), eager[1].view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(captured[2], eager[2], rtol=0, atol=0)
            torch.testing.assert_close(captured[3], eager[3], rtol=0, atol=0)
            if last_scale is not None:
                self.assertFalse(torch.equal(captured[2], last_scale))
            last_scale = captured[2].clone()


if __name__ == "__main__":
    unittest.main()
