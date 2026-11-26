"""HCU compatibility hooks for shared SGLang kernel helpers."""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def after_get_default_target_flags(flags: list[str], arch: Any = None) -> list[str]:
    if "-DUSE_ROCM" not in flags:
        return flags
    fp8_flag = "-DHIP_FP8_TYPE_E4M3=1"
    try:
        device = torch.cuda.current_device()
        gcn_arch = torch.cuda.get_device_properties(device).gcnArchName
        if any(gfx in gcn_arch for gfx in (
            "gfx90a", "gfx936", "gfx938", "gfx940", "gfx941", "gfx942"
        )):
            fp8_flag = "-DHIP_FP8_TYPE_FNUZ=1"
    except Exception:
        pass
    return [
        fp8_flag if flag in ("-DHIP_FP8_TYPE_E4M3=1", "-DHIP_FP8_TYPE_FNUZ=1") else flag
        for flag in flags
    ]


def around_assign_extend_cache_locs_func(
    original: Callable[..., torch.Tensor],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device: Any,
) -> torch.Tensor:
    if not is_hcu():
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            batch_size,
            draft_token_num,
            device,
        )

    native = importlib.import_module("sglang.kernels.ops.speculative.cache_locs")
    platform_fn = native._get_oot_speculative_cache_locs_fn()
    if platform_fn is not None:
        return platform_fn(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            start_offset=start_offset,
            end_offset=end_offset,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            device=device,
        )

    from sglang.srt.utils import get_bool_env_var

    gather = getattr(torch.ops.sgl_kernel, "hcu_assign_extend_cache_locs", None)
    if gather is None or not get_bool_env_var("SGLANG_ASSIGN_EXTEND_CACHE_LOCS", default="true"):
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            batch_size,
            draft_token_num,
            device,
        )

    out_cache_loc = torch.empty(
        (batch_size * draft_token_num,), dtype=torch.int64, device=device
    )
    gather(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        batch_size,
    )
    return out_cache_loc


def _dispatch_mhc(name: str, original: Callable[..., Any], *args: Any, **kwargs: Any):
    if not is_hcu():
        return original(*args, **kwargs)
    from hcu_sglang.kernels import mhc

    return getattr(mhc, name)(*args, **kwargs)


def around_hc_split_sinkhorn(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("hc_split_sinkhorn", original, *args, **kwargs)


def around_prewarm_mhc_pre(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("prewarm_mhc_pre", original, *args, **kwargs)


def around_mhc_pre(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("mhc_pre", original, *args, **kwargs)


def around_mhc_post(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("mhc_post", original, *args, **kwargs)


@functools.lru_cache(maxsize=1)
def _native_mhc_functions():
    from aiter.ops.tilelang import mhc_post_fwd, mhc_pre_big_fuse

    logging.getLogger(__name__).info(
        "HCU AITER TileLang mHC enabled; output normalization remains with the caller"
    )
    return mhc_pre_big_fuse, mhc_post_fwd


def _aiter_tilelang_mhc_enabled(use_tilelang: bool) -> bool:
    from sglang.srt.environ import envs

    return (
        use_tilelang
        and is_hcu()
        and envs.SGLANG_ROCM_USE_AITER_TILELANG_MHC.get()
    )


def around_native_mhc_pre(
    original, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    norm_weight=None, norm_eps=None,
):
    from sglang.srt.environ import envs

    if not _aiter_tilelang_mhc_enabled(envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()):
        return original(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            norm_weight, norm_eps,
        )
    pre, _ = _native_mhc_functions()
    post, comb, output = pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=16,
    )
    return post, comb, output, False


def around_native_mhc_post(original, x, residual, post_layer_mix, comb_res_mix):
    from sglang.srt.environ import envs

    if not _aiter_tilelang_mhc_enabled(envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()):
        return original(x, residual, post_layer_mix, comb_res_mix)
    _, post = _native_mhc_functions()
    return post(x, residual, post_layer_mix.squeeze(-1), comb_res_mix)


def around_mhc_fused_post_pre(
    original: Callable[..., Any], *args: Any, **kwargs: Any
):
    return _dispatch_mhc("mhc_fused_post_pre", original, *args, **kwargs)


def around_vision_flash3_init(
    original: Callable[..., Any], instance: Any, *args: Any, **kwargs: Any
):
    if not is_hcu():
        return original(instance, *args, **kwargs)
    vision = importlib.import_module("sglang.srt.layers.attention.vision")
    from hcu_sglang.layers.attention.flashattention_interface import (
        flash_attn_varlen_func,
    )

    vision.flash_attn_func = flash_attn_varlen_func
    previous_is_cuda = vision._is_cuda
    vision._is_cuda = True
    try:
        return original(instance, *args, **kwargs)
    finally:
        vision._is_cuda = previous_is_cuda


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.kernels.jit.utils.arch.get_default_target_flags",
        after_get_default_target_flags,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.cache_locs."
        "assign_extend_cache_locs_func",
        around_assign_extend_cache_locs_func,
        HookType.AROUND,
    )
    mhc_target = "sglang.kernels.ops.layernorm.mhc"
    for name, hook in (
        ("_mhc_pre_dispatch", around_native_mhc_pre),
        ("_mhc_post_dispatch", around_native_mhc_post),
        ("hc_split_sinkhorn", around_hc_split_sinkhorn),
        ("prewarm_mhc_pre", around_prewarm_mhc_pre),
        ("mhc_pre", around_mhc_pre),
        ("mhc_post", around_mhc_post),
        ("mhc_fused_post_pre", around_mhc_fused_post_pre),
    ):
        HookRegistry.register(f"{mhc_target}.{name}", hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.layers.attention.vision.VisionFlash3Attention.__init__",
        around_vision_flash3_init,
        HookType.AROUND,
    )
