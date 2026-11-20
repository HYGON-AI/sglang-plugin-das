import logging
import os
from functools import cache, partial

import torch

from .minimax_m31_shared_overlap import resolve_single_dispatcher


LOG = logging.getLogger(__name__)


@cache
def _announce():
    LOG.info("MiniMax CP Prefill shared expert overlap active (split/side-stream)")


@cache
def _lightop_swiglu_int8():
    try:
        from lightop.activation import swiglu_oai_int8_quant as ops
    except (ImportError, OSError):
        return None
    if getattr(ops, "_native", None) is None:
        return None
    return getattr(ops, "swiglu_oai_int8_quant_contiguous", None)


@cache
def _announce_swiglu_int8():
    LOG.info("MiniMax CP Prefill shared expert uses LightOp OAI INT8 quantization")


def _shared_int8_kernel(shared):
    down = shared.down_proj
    if (
        not shared.fuse_swiglu_oai_mxfp8
        or not down.input_is_parallel
        or down.parallel_group != "replicated"
        or down.input_size_per_partition != 3072
        or down.bias is not None
        or down.reduce_results
        or down.use_dp_attention_reduce
        or down.use_decode_attn_tp
        or not getattr(down.quant_method, "supports_prequantized_input", False)
    ):
        return None
    return _lightop_swiglu_int8()


def _shared_down(shared, gate_up, quantize):
    if (
        quantize is not None
        and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.shape[0] > 0
        and gate_up.shape[1] == 6144
        and gate_up.is_contiguous()
    ):
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric
        from sglang.srt.runtime_context import get_parallel

        q, scales = quantize(gate_up, shared.swiglu_alpha, 1.0, shared.swiglu_limit)
        down = shared.down_proj
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            output = down.quant_method.apply(
                down, gate_up[:, : q.shape[1]], bias=None, input_quant_args=(q, scales)
            )
        _announce_swiglu_int8()
        return output

    activated = shared._swiglu_oai_mxfp8(gate_up)
    if activated is None:
        activated = shared.act_fn(gate_up)
    output, _ = shared.down_proj(activated)
    return output


def _forward_deepep(original, moe, hidden_states, forward_batch, *, split, use_lightop):
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_forward, get_parallel, get_stream

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
        or (split and not dispatcher._normal_dispatcher.async_finish)
        or torch.cuda.is_current_stream_capturing()
    ):
        return original(moe, hidden_states, forward_batch)

    shared = moe.shared_experts
    quantize = _shared_int8_kernel(shared) if use_lightop and not get_forward().sp_active else None
    if not split:
        if quantize is None:
            return original(moe, hidden_states, forward_batch)
        logits = moe._compute_router_logits(hidden_states)
        gate_up, _ = shared.gate_up_proj(hidden_states)
        shared_output = _shared_down(shared, gate_up, quantize)
        routed = moe.experts(hidden_states, moe.topk(hidden_states, logits))
        return routed + shared_output

    _announce()
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
            output = _shared_down(shared, gate_up, quantize)

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
    use_oai_int8 = os.getenv("SGLANG_MINIMAX_PREFILL_SHARED_OAI_INT8", "1") == "1"
    if method == "off" and not use_oai_int8:
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.minimax_m3.MiniMaxM3MoE.forward_deepep",
        partial(_forward_deepep, split=method == "split", use_lightop=use_oai_int8),
        HookType.AROUND,
    )
