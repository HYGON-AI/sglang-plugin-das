"""HCU single-batch-overlap enablement."""

from __future__ import annotations

from typing import Callable

from hcu_sglang.utils.detection import is_hcu


def around_enable_combine_down_gemm(original: Callable[..., bool], cls: type) -> bool:
    if is_hcu():
        from sglang.srt.layers.moe.utils import is_sbo_enabled

        return is_sbo_enabled()
    return original(cls)


def around_compute_overlap_args(original, dispatch_output, alt_stream):
    if is_hcu():
        import torch

        hidden_states = dispatch_output.hidden_states
        if hidden_states.dtype in (
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ) or hidden_states.ndim != 3:
            return None, None, {}
    return original(dispatch_output, alt_stream)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap."
        "SboFlags.enable_combine_down_gemm_two_stream_overlap",
        around_enable_combine_down_gemm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap.compute_overlap_args",
        around_compute_overlap_args,
        HookType.AROUND,
    )
