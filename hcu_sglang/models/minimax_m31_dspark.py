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


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.speculative_hook._handle_dspark",
        _resolve_dspark,
        HookType.AROUND,
    )
