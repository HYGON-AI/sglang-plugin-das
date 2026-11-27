"""HCU single-batch-overlap enablement."""

from __future__ import annotations

from typing import Callable

from hcu_sglang.utils.detection import is_hcu


def _hcu_moe_is_cuda() -> bool:
    """Report the physical backend to the MoE SM90 overlap guard."""
    return False


def around_enable_combine_down_gemm(original: Callable[..., bool], cls: type) -> bool:
    if is_hcu():
        from sglang.srt.layers.moe.utils import is_sbo_enabled

        return is_sbo_enabled()
    return original(cls)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if is_hcu():
        from sglang.srt.layers.moe import utils as moe_utils

        # Keep the override in this module. HookRegistry propagates wrapped
        # functions to modules that imported the same detection predicate.
        moe_utils.is_cuda = _hcu_moe_is_cuda
    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap."
        "SboFlags.enable_combine_down_gemm_two_stream_overlap",
        around_enable_combine_down_gemm,
        HookType.AROUND,
    )
