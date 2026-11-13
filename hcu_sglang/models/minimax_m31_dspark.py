from functools import cache
from types import FunctionType

_BASE_BACKENDS = ("none", "megamoe", "mori")
_MINIMAX_ARCHES = {"MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM"}


@cache
def _with_deepep(original):
    code = original.__code__
    if code.co_consts.count(_BASE_BACKENDS) != 1:
        raise RuntimeError("v0.5.23 DSpark DP MoE backend check has changed")
    constants = tuple(
        (*value, "deepep") if value == _BASE_BACKENDS else value
        for value in code.co_consts
    )
    return FunctionType(
        code.replace(co_consts=constants),
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def _resolve_dspark(original, server_args):
    from sglang.srt.arg_groups.overrides import model_config_of, resolving_view

    cfg = resolving_view(server_args)
    if (
        cfg.moe_a2a_backend != "deepep"
        or cfg.attn_dp_size <= 1
        or cfg.quantization != "slimquant_w4a8_marlin"
    ):
        return original(server_args)

    import torch

    arch = model_config_of(server_args).hf_config.architectures[0]
    if arch not in _MINIMAX_ARCHES or not torch.version.hip:
        return original(server_args)
    return _with_deepep(original)(server_args)


def _minimax_hip_dspark():
    import torch
    from sglang.srt.runtime_context import get_model, get_spec

    return (
        torch.version.hip is not None
        and get_spec().speculative_algorithm == "DSPARK"
        and get_model().quantization == "slimquant_w4a8_marlin"
    )


@cache
def _with_native_hip(original):
    return FunctionType(
        original.__code__,
        {**original.__globals__, "_is_cuda": False, "_is_hip": True},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def _hip_future_map(original, relay, *args, **kwargs):
    # The private D2H copy stream is CUDA-only in the community relay.
    # Use its native HIP publish fence and synchronous CPU length read.
    if _minimax_hip_dspark():
        original = _with_native_hip(original)
    return original(relay, *args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.speculative_hook._handle_dspark",
        _resolve_dspark,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.managers.overlap_utils.FutureMap.__init__",
        _hip_future_map,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.overlap_utils.FutureMap.resolve_seq_lens_cpu",
        _hip_future_map,
        HookType.AROUND,
    )
