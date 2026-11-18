"""Quantized MTP draft lm_head.

The draft head is the widest GEMM of a decode step (248320 x 5120); its bf16
copy streams ~2.37 GiB per call, and the quantized copy plus the packed tiles
take it from 1320 to 792 us at M=45. Only the draft model's lm_head is
replaced, so the target head stays bf16.

The head is fp8 e4m3 with a per-row scale; the GEMM backend that serves the
result lives in :mod:`hcu_sglang.layers.gemm.quant_lm_head`.
"""

from __future__ import annotations

import logging

import torch

# Per-row block the draft head is quantized in.
_BLOCK = 4096

# e4m3's largest finite magnitude; the same value scales weights and rows.
_FP8_MAX = 448.0

# amax floor used when scaling fp8 activations and weights.
_AMAX_EPS = 1e-8

logger = logging.getLogger(__name__)


def _quantize_rows(weight: torch.Tensor):
    vocab, hidden = weight.shape
    weight_q = torch.empty(
        (vocab, hidden), dtype=torch.float8_e4m3fn, device=weight.device
    )
    weight_scale = torch.empty(
        (vocab, 1), dtype=torch.float32, device=weight.device
    )
    for i in range(0, vocab, _BLOCK):
        j = min(i + _BLOCK, vocab)
        wf = weight[i:j].to(torch.float32)
        scale = (
            wf.abs().amax(dim=-1, keepdim=True).clamp_min(_AMAX_EPS) / _FP8_MAX
        )
        weight_q[i:j] = (wf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(
            torch.float8_e4m3fn
        )
        weight_scale[i:j] = scale
        del wf
    return weight_q, weight_scale


def _quantize_x(x: torch.Tensor):
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(_AMAX_EPS) / _FP8_MAX
    xq = (xf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return xq, scale


class DraftHeadQuantMethod:
    def __init__(self, base, weight_q, weight_scale, gemm, max_m):
        self.base = base
        self.weight_q = weight_q
        self.weight_scale = weight_scale
        self.gemm = gemm
        self.max_m = max_m
        # One buffer for every decode row count: a per-call allocation lands in
        # the CUDA-graph pool once per bs bucket (~0.4 GiB over bs 1..40) and
        # that is what pushed verify into OOM. Different bs buckets are captured
        # into different graphs and never run concurrently, so the shared buffer
        # does not alias across them; apply slices it per call.
        self.out = torch.empty(
            (max_m, weight_q.shape[0]),
            dtype=torch.bfloat16,
            device=weight_q.device,
        )

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        assert bias is None, "quantized draft lm_head does not support a bias"
        if x.shape[0] > self.max_m:
            # ParallelLMHead.forward raises by design (the head's weight belongs
            # to the sampler), so wide rows go through the weight directly.
            return torch.matmul(x, self.base.weight.t())
        xq, xs = _quantize_x(x)
        out = self.out[: x.shape[0]]
        self.gemm(xq, self.weight_q, xs, self.weight_scale, out=out)
        return out


def quantize(weight: torch.Tensor):
    """The quantized head and its per-row scale, or None to keep bf16."""
    try:
        weight_q, weight_scale = _quantize_rows(weight)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.warning(
            "[gemm] disabling, quantization failed: %s", exc
        )
        return None
    return weight_q, weight_scale
