from functools import cache

import torch

from hcu_sglang.layers.moe.minimax_router_gemv import (
    can_use_minimax_router_gemv,
    minimax_router_gemv,
)


@cache
def _is_gfx938(device):
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "gcnArchName", "").split(":")[0] == "gfx938"


@cache
def _lightop_router():
    try:
        from lightop import _router_gemm
    except ImportError:
        return None
    return _router_gemm


def _compute_router_logits(original, layer, hidden_states):
    weight = layer.gate.weight
    if (
        not layer.bf16_router_gemm
        or not torch.version.hip
        or not can_use_minimax_router_gemv(hidden_states, weight)
        or hidden_states.device != weight.device
        or not _is_gfx938(hidden_states.device)
    ):
        return original(layer, hidden_states)
    if (
        hidden_states.shape[0] == 64
        and hidden_states.storage_offset() % 8 == 0
        and weight.storage_offset() % 8 == 0
    ):
        native = _lightop_router()
        if native is not None:
            return native.forward(hidden_states, weight)
    return minimax_router_gemv(hidden_states, weight, block_k=1024, num_warps=1)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.minimax_m3.MiniMaxM3MoE._compute_router_logits",
        _compute_router_logits,
        HookType.AROUND,
    )
