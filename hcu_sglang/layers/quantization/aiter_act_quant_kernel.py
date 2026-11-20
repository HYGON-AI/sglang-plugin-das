import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _silu_mul_int8(X, Q, S, M, N: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    gate = tl.load(X + rows[:, None] * (2 * N) + cols[None, :], mask, 0).to(tl.float32)
    up = tl.load(X + rows[:, None] * (2 * N) + N + cols[None, :], mask, 0).to(tl.float32)
    exponent = tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", constraints="=v,v", args=[-gate * 1.4426950408889634],
        dtype=tl.float32, is_pure=True, pack=1)
    reciprocal = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", constraints="=v,v", args=[1.0 + exponent],
        dtype=tl.float32, is_pure=True, pack=1)
    activated = (gate * reciprocal).to(tl.bfloat16).to(tl.float32)
    product = (activated * up).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(product), axis=1), 1e-10)
    scale = maximum / 127.0
    codes = libdevice.nearbyint(product * (127.0 / maximum[:, None])).to(tl.int8)
    tl.store(Q + rows[:, None] * N + cols[None, :], codes, mask)
    tl.store(S + rows, scale, rows < M)


def silu_mul_int8(input, dtype=torch.int8, *, block_m=8, num_warps=4):
    if (input.ndim != 2 or input.dtype != torch.bfloat16
            or not input.is_contiguous() or dtype != torch.int8 or input.shape[1] % 2):
        raise ValueError("AITER fused activation requires contiguous BF16 gate/up and INT8 output")
    m, twice_n = input.shape
    n = twice_n // 2
    q = torch.empty((m, n), device=input.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=input.device, dtype=torch.float32)
    if m:
        _silu_mul_int8[(triton.cdiv(m, block_m),)](
            input, q, s, m, n, block_m, triton.next_power_of_2(n),
            num_warps=num_warps, enable_fp_fusion=False)
    return q, s
