# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase, main, skipUnless

import torch
from lightop.quant import per_token_quant_int8
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

from hcu_sglang.layers.attention.sigmoid_mul_int8 import sigmoid_mul_per_token_int8_quant


@skipUnless(torch.cuda.is_available(), "GPU required")
class SigmoidMulInt8Tests(TestCase):
    def assert_matches_reference(self, attn, gate):
        baseline_gate = gate.contiguous() if gate.ndim == 2 else gate
        gated = fused_sigmoid_mul(attn, baseline_gate)
        expected = per_token_quant_int8(gated)
        actual = sigmoid_mul_per_token_int8_quant(attn, gate)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_matches_strided_qwen_gate(self):
        torch.manual_seed(42)
        for tokens in (1, 4, 16, 128, 512, 2048):
            with self.subTest(tokens=tokens):
                attn = torch.randn((tokens, 3072), device="cuda", dtype=torch.bfloat16)
                gate = torch.randn((tokens, 12, 512), device="cuda", dtype=attn.dtype)[..., 256:]
                self.assert_matches_reference(attn, gate)

    def test_flat_strided_gate_and_zero_row(self):
        for width in (640, 3072):
            with self.subTest(width=width):
                attn = torch.randn((4, width), device="cuda", dtype=torch.bfloat16)
                attn[0].zero_()
                gate = torch.randn((4, 2 * width), device="cuda", dtype=attn.dtype)[:, width:]
                self.assert_matches_reference(attn, gate)


    def test_empty_and_invalid_gate(self):
        attn = torch.empty((0, 3072), device="cuda", dtype=torch.bfloat16)
        q, scale = sigmoid_mul_per_token_int8_quant(attn, attn.reshape(0, 12, 256))
        self.assertEqual(tuple(q.shape), (0, 3072))
        self.assertEqual(tuple(scale.shape), (0, 1))
        with self.assertRaisesRegex(ValueError, "Gate shape"):
            sigmoid_mul_per_token_int8_quant(attn, attn[:, :1])

    def test_graph_replay_with_updated_input(self):
        attn = torch.randn((4, 3072), device="cuda", dtype=torch.bfloat16)
        gate = torch.randn((4, 12, 512), device="cuda", dtype=attn.dtype)[..., 256:]
        sigmoid_mul_per_token_int8_quant(attn, gate)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sigmoid_mul_per_token_int8_quant(attn, gate)
        for value in (0.25, -1.5):
            attn.fill_(value)
            gate.normal_()
            graph.replay()
            expected = per_token_quant_int8(fused_sigmoid_mul(attn, gate))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    main()
