from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _round_even(x):
    lower = tl.floor(x)
    fraction = x - lower
    return lower + ((fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))).to(tl.float32)


@triton.jit
def rope_fp4_kernel(X, F, Y, rows, heads: tl.constexpr, sx0: tl.constexpr,
                    sx1: tl.constexpr, sx2: tl.constexpr, sf0: tl.constexpr,
                    sf1: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                    GROUP: tl.constexpr, COMPRESSED: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, D)
    base = (row // heads) * sx0 + (row % heads) * sx1
    x = tl.load(X + base[:, None] + d[None, :] * sx2, row[:, None] < rows, 0).to(tl.float32)
    partner = tl.load(X + base[:, None] + (d[None, :] ^ 1) * sx2, row[:, None] < rows, 0).to(tl.float32)
    pair = tl.maximum(d - (D - R), 0) // 2
    real = tl.load(F + (row // heads)[:, None] * sf0 + (2 * pair)[None, :] * sf1,
                   (row[:, None] < rows) & (d[None, :] >= D - R), 1)
    imag = tl.load(F + (row // heads)[:, None] * sf0 + (2 * pair + 1)[None, :] * sf1,
                   (row[:, None] < rows) & (d[None, :] >= D - R), 0)
    rotated = x * real + partner * tl.where((d[None, :] & 1) == 0, -imag, imag)
    value = tl.where(d[None, :] >= D - R, rotated.to(tl.bfloat16).to(tl.float32), x)
    blocks = tl.reshape(value, (B, D // GROUP, GROUP))
    peak = tl.max(tl.abs(blocks), 2)
    peak = tl.where(tl.sum((blocks != blocks).to(tl.int32), 2) > 0, float("nan"), peak)
    if COMPRESSED:
        s = peak * (1.0 / 6.0)
        s = tl.where(s != s, s, tl.minimum(tl.maximum(s, 0.001953125), 448.0))
        bits = s.to(tl.int32, bitcast=True)
        exponent = tl.maximum(((bits >> 23) & 255) - 127 - 3, -9)
        quantum = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        scale = _round_even(s / quantum) * quantum
    else:
        peak = tl.where(peak != peak, peak, tl.maximum(peak, 6.0 * 1.1754943508222875e-38))
        s = peak * (1.0 / 6.0)
        bits = s.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 8388607) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
    scaled = blocks / scale[:, :, None]
    scaled = tl.where(scaled != scaled, scaled, tl.minimum(tl.maximum(scaled, -6.0), 6.0))
    magnitude = tl.abs(scaled)
    step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    result = _round_even(magnitude / step) * step * sign * scale[:, :, None]
    tl.store(Y + row[:, None] * D + d[None, :], tl.reshape(result, (B, D)), row[:, None] < rows)


@lru_cache(maxsize=8)
def _architecture(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0]


def supported(x, freqs, rope_dim):
    return (
        x.is_cuda and x.dtype == torch.bfloat16 and x.ndim in (2, 3)
        and x.shape[-1] in (128, 512) and rope_dim == 64
        and freqs.dtype == torch.complex64 and freqs.shape == (x.shape[0], rope_dim // 2)
        and freqs.device == x.device and _architecture(x.device.index) == "gfx936"
    )


def rope_fp4_fake_quant(x, freqs, rope_dim, compressed_kv=False):
    if not supported(x, freqs, rope_dim):
        raise ValueError("Unsupported HCU fused RoPE/FP4 tensors")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    heads = x.shape[1] if x.ndim == 3 else 1
    rows = x.shape[0] * heads
    if not rows:
        return output
    f = torch.view_as_real(freqs).flatten(-2)
    rope_fp4_kernel[(triton.cdiv(rows, 4),)](
        x, f, output, rows, heads, x.stride(0), x.stride(1) if x.ndim == 3 else 0,
        x.stride(-1), f.stride(0), f.stride(1), x.shape[-1], rope_dim,
        16 if compressed_kv else 32, compressed_kv, 4, num_warps=4, enable_fp_fusion=False,
    )
    return output
