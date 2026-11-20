import logging
import os
from functools import cache

import torch

from .minimax_m31_shared_overlap import resolve_single_dispatcher


LOG = logging.getLogger(__name__)


@cache
def _announce():
    LOG.info("MiniMax CP Prefill shared expert overlap active (split/side-stream)")


def _forward_deepep(original, moe, hidden_states, forward_batch):
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_parallel, get_stream

    dispatcher = resolve_single_dispatcher(moe.experts.dispatcher)
    if (
        not torch.version.hip
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL)
        or hidden_states.shape[0] == 0
        or moe.shared_experts is None
        or moe.num_fused_shared_experts
        or not getattr(moe.experts, "_minimax_w4a8_hipc", False)
        or get_parallel().attn_cp_size <= 1
        or is_sbo_enabled()
        or is_tbo_enabled()
        or dispatcher is None
        or not dispatcher.deepep_mode.is_normal()
        or not dispatcher._normal_dispatcher.async_finish
        or torch.cuda.is_current_stream_capturing()
    ):
        return original(moe, hidden_states, forward_batch)

    _announce()
    shared = moe.shared_experts
    main_stream = torch.cuda.current_stream(hidden_states.device)
    shared_stream = get_stream("minimax_m31_prefill_shared")
    shared_stream.wait_stream(main_stream)
    with torch.cuda.stream(shared_stream):
        gate_up, _ = shared.gate_up_proj(hidden_states)
    output = None

    def project_down(_dispatcher):
        nonlocal output
        shared_stream.wait_stream(main_stream)
        with torch.cuda.stream(shared_stream):
            activated = shared._swiglu_oai_mxfp8(gate_up)
            if activated is None:
                activated = shared.act_fn(gate_up)
            output, _ = shared.down_proj(activated)

    topk = moe.topk(hidden_states, moe._compute_router_logits(hidden_states))
    handle = dispatcher.register_deepep_normal_combine_hook(project_down)
    try:
        routed = moe.experts(hidden_states, topk)
        if output is None:
            project_down(dispatcher)
    finally:
        handle.remove()
        main_stream.wait_stream(shared_stream)

    return routed + output


def register_hooks():
    method = os.getenv("SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD", "off")
    if method not in ("off", "split"):
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD must be off or split"
        )
    if method == "off":
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.minimax_m3.MiniMaxM3MoE.forward_deepep",
        _forward_deepep,
        HookType.AROUND,
    )
