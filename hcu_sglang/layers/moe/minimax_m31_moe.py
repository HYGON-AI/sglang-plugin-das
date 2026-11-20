from functools import cache
from types import FunctionType

import torch

from hcu_sglang.layers.quantization.swiglu_m31_int8 import swiglu_oai_int8_quant


@cache
def _hip_init(original):
    return FunctionType(
        original.__code__,
        {**original.__globals__, "_is_hip": True},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def init_on_hip(original, layer, args, kwargs):
    config = kwargs.get("quant_config")
    if not torch.version.hip or config is None or config.get_name() != "slimquant_w4a8_marlin":
        return original(layer, *args, **kwargs)
    return _hip_init(original)(layer, *args, **kwargs)


def configure_dispatcher(layer):
    layer.dispatcher.set_quant_config({
        "normal_dispatcher_output_dtype": "int8",
        "low_latency_dispatcher_output_dtype": "int8",
        "normal_expert_alignment": 256,
        "hcu_w4a8_int8_dispatch": True,
        "hcu_w4a8_deepep_ll_producer_event": True,
    })
    layer._minimax_w4a8_hipc = True


def _activate(layer, x, counts=None):
    config = layer.moe_runner_config
    return swiglu_oai_int8_quant(
        x, alpha=config.gemm1_alpha, beta=config.gemm1_beta, limit=config.gemm1_clamp_limit,
        tokens_per_expert=counts, allow_undefined_padding=counts is not None,
    )


def _normal(layer, dispatch):
    from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as gemm
    from hcu_sglang.layers.moe.deepep_fp8 import (
        _ep_gather_with_optional_lightop,
        _ep_scatter_with_optional_lightop,
        copy_list_to_gpu_no_ce,
        get_offloader,
    )

    x, scale, ids, weights, recv = dispatch
    rows = sum(recv) if recv else 0
    if rows == 0:
        return x.bfloat16()
    if x.dtype != torch.int8 or scale is None or any(n % 256 for n in recv):
        raise RuntimeError("MiniMax HIPC needs INT8 DeepEP dispatch and expert counts aligned to 256")
    if get_offloader().forbid_copy_engine_usage:
        counts = copy_list_to_gpu_no_ce(recv)
    else:
        counts = torch.tensor(recv, dtype=torch.int32, pin_memory=True).to(x.device, non_blocking=True)
    reordered = x.new_empty((rows, x.shape[1]))
    scales = scale.new_empty((rows, 1), dtype=torch.float32)
    expert_ids, gather_index = _ep_scatter_with_optional_lightop(
        x, scale, ids, counts, reordered, scales, rows, counts_are_aligned=True,
    )
    gate_up = x.new_empty((rows, layer.w13_weight_scale.shape[1]), dtype=torch.bfloat16)
    gemm((reordered, scales), (layer.w13_weight, layer.w13_weight_scale), gate_up, expert_ids)
    act = _activate(layer, gate_up)
    down = x.new_empty((rows, x.shape[1]), dtype=torch.bfloat16)
    gemm(act, (layer.w2_weight, layer.w2_weight_scale), down, expert_ids)
    output = x.new_empty(x.shape, dtype=torch.bfloat16)
    _ep_gather_with_optional_lightop(down, ids, weights, gather_index, output)
    return output


def _low_latency(layer, dispatch):
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc as gemm

    x, scale, _, _, counts, expected = dispatch
    if x.dtype != torch.int8 or scale is None:
        raise RuntimeError("MiniMax masked HIPC needs INT8 DeepEP dispatch and per-token scales")
    groups, rows, hidden = x.shape
    expected = min(rows, expected)
    gate_up = x.new_empty((groups, rows, layer.w13_weight_scale.shape[1]), dtype=torch.bfloat16)
    gemm((x, scale), (layer.w13_weight, layer.w13_weight_scale), gate_up, counts, expected)
    act = _activate(layer, gate_up, counts)
    output = x.new_empty((groups, rows, hidden), dtype=torch.bfloat16)
    gemm(act, (layer.w2_weight, layer.w2_weight_scale), output, counts, expected)
    return output


def run_core(layer, dispatch):
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    if DispatchOutputChecker.format_is_deepep_normal(dispatch):
        return DeepEPNormalCombineInput(_normal(layer, dispatch), dispatch.topk_ids, dispatch.topk_weights)
    if DispatchOutputChecker.format_is_deepep_ll(dispatch):
        return DeepEPLLCombineInput(_low_latency(layer, dispatch), dispatch.topk_ids, dispatch.topk_weights)
    raise ValueError("MiniMax HIPC requires DeepEP normal or low-latency dispatch")
