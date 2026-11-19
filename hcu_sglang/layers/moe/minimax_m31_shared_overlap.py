import os

import torch


def _shared_in_receive_windows(moe, hidden_states):
    shared = moe.shared_experts
    dispatcher = moe.experts.dispatcher
    low_latency = dispatcher._low_latency_dispatcher
    gate_up = shared_output = None

    def project_gate_up(_dispatcher):
        nonlocal gate_up
        gate_up, _ = shared.gate_up_proj(hidden_states)

    def project_down(_dispatcher):
        nonlocal shared_output
        x = shared._swiglu_oai_mxfp8(gate_up)
        if x is None:
            x = shared.act_fn(gate_up)
        shared_output, _ = shared.down_proj(x)

    topk = moe.topk(hidden_states, moe._compute_router_logits(hidden_states))
    recv_hook = low_latency.return_recv_hook
    low_latency.return_recv_hook = True
    handles = []
    try:
        handles.extend(dispatcher.register_deepep_combine_hook(project_down))
        handles.extend(dispatcher.register_deepep_dispatch_hook(project_gate_up))
        routed = moe.experts(hidden_states, topk)
        if shared_output is None:
            raise RuntimeError("DeepEP did not complete the MiniMax shared expert projections")
        return routed + shared_output
    finally:
        for handle in handles:
            handle.remove()
        low_latency.return_recv_hook = recv_hook


def _forward_deepep(original, moe, hidden_states, forward_batch):
    if (
        os.getenv("SGLANG_MINIMAX_DECODE_SHARED_OVERLAP", "1") != "1"
        or not torch.version.hip
        or hidden_states.shape[0] == 0
        or moe.shared_experts is None
        or moe.num_fused_shared_experts
        or not getattr(moe.experts, "_minimax_w4a8_hipc", False)
        or not (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
    ):
        return original(moe, hidden_states, forward_batch)

    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.runtime_context import get_memory, get_parallel

    from .deepep_current import DeepEPDispatcher

    parallel = get_parallel()
    dispatcher = moe.experts.dispatcher
    if (
        get_memory().enable_hisparse
        or parallel.attn_dp_size != 8
        or parallel.moe_ep_size != 8
        or is_sbo_enabled()
        or is_tbo_enabled()
        or not isinstance(dispatcher, DeepEPDispatcher)
        or not dispatcher.deepep_mode.enable_low_latency()
        or dispatcher._get_impl() is not dispatcher._low_latency_dispatcher
    ):
        return original(moe, hidden_states, forward_batch)
    return _shared_in_receive_windows(moe, hidden_states)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.minimax_m3.MiniMaxM3MoE.forward_deepep",
        _forward_deepep,
        HookType.AROUND,
    )
