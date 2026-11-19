# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.utils.detection import is_hcu


def _tilelang_enabled():
    return is_hcu() and envs.SGLANG_OPT_HY4_IHC_TILELANG.get()


def around_ihc_pre(original, layer, hidden_states, rms_weight=None, rms_eps=0.0):
    if not _tilelang_enabled():
        return original(layer, hidden_states, rms_weight, rms_eps)
    from hcu_sglang.models.hyv4_ihc_tilelang import try_tilelang_ihc_pre

    result = try_tilelang_ihc_pre(
        hidden_states, layer.hc_fn.weight, layer.hc_scale, layer.hc_base,
        layer.rms_norm_eps, layer.hc_eps, layer.magnitude,
    )
    if result is None:
        return original(layer, hidden_states, rms_weight, rms_eps)
    reduced, post = result
    if rms_weight is not None:
        import torch
        value = reduced.float()
        reduced = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + rms_eps)
            * rms_weight.float()
        ).to(reduced.dtype)
    return reduced, post


def around_ihc_post(original, layer, output, residual, post):
    if not _tilelang_enabled():
        return original(layer, output, residual, post)
    from hcu_sglang.models.hyv4_ihc_tilelang import try_tilelang_ihc_post

    result = try_tilelang_ihc_post(output, residual, post)
    return original(layer, output, residual, post) if result is None else result


def around_ihc_post_pre(original, layer, output, residual, post, next_layer, norm):
    if not _tilelang_enabled():
        return original(layer, output, residual, post, next_layer, norm)
    next_residual = layer.post(output, residual, post)
    return next_layer.pre(next_layer.prepare_input(next_residual), norm)


def around_ihc_head(original, layer, hidden_states, norm=None):
    if not _tilelang_enabled():
        return original(layer, hidden_states, norm)
    from hcu_sglang.models.hyv4_ihc_tilelang import try_tilelang_ihc_head

    result = try_tilelang_ihc_head(
        hidden_states, layer.hc_head_fn.weight, layer.hc_head_scale,
        layer.hc_head_base, layer.config.rms_norm_eps, layer.config.hc_eps,
    )
    if result is None:
        return original(layer, hidden_states, norm)
    return result if norm is None else norm(result)


def around_attention_init(original, layer, config, *args, **kwargs):
    result = original(layer, config, *args, **kwargs)
    if not (is_hcu() and envs.SGLANG_HY4_LINEAR_GATE_CP_SHARD.get()):
        return result
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from hcu_sglang.models.hyv4_gate_cp_shard import get_gate_cp_shard_runtime

    parallel = get_parallel()
    width = layer.local_gate_width
    if not (
        parallel.enable_prefill_cp and parallel.attn_cp_size > 1
        and parallel.attn_tp_size == 1 and parallel.attn_dcp_size == 1
        and width % parallel.attn_cp_size == 0
    ):
        raise ValueError("HYV4 CP-sharded gate requires prefill CP, attention TP1 and DCP1")
    if not isinstance(layer.linear_gate.quant_method, UnquantizedLinearMethod):
        raise ValueError("HYV4 CP-sharded gate requires an unquantized gate projection")
    if parallel.tp_group.ranks != parallel.attn_cp_group.ranks:
        raise ValueError("HYV4 CP-sharded gate requires matching TP and attention CP partitions")
    prefix = kwargs.get("prefix", args[2] if len(args) > 2 else "")
    layer.linear_gate = ColumnParallelLinear(
        config.hidden_size, width, bias=False, quant_config=None,
        parallel_group="tp",
        prefix=f"{prefix}.linear_gate",
    )
    layer._gate_cp_runtime = get_gate_cp_shard_runtime(layer.linear_gate.weight)
    layer.gate_cp_shard_enabled = True
    layer._gate_cp_chain = None
    layer._gate_cp_index = None
    layer._gate_backend = layer._gate_fallback_backend
    return result


def around_model_init(original, model, *args, **kwargs):
    result = original(model, *args, **kwargs)
    if is_hcu() and envs.SGLANG_HY4_LINEAR_GATE_CP_SHARD.get():
        from hcu_sglang.models.hyv4_gate_cp_shard import build_gate_cp_chain

        model.gate_cp_chain = build_gate_cp_chain(
            [layer.self_attn for layer in model.layers]
        )
    return result


def around_model_forward(original, model, input_ids, positions, forward_batch, input_embeds=None):
    chain = getattr(model, "gate_cp_chain", None)
    if chain is None:
        return original(model, input_ids, positions, forward_batch, input_embeds=input_embeds)
    from sglang.srt.layers.cp.utils import is_cp_active

    num_tokens = len(input_embeds) if input_embeds is not None else len(input_ids)
    chain.begin(is_cp_active(forward_batch), num_tokens)
    try:
        return original(model, input_ids, positions, forward_batch, input_embeds=input_embeds)
    finally:
        chain.end()


def around_prepare_gate(original, layer, hidden_states):
    if not getattr(layer, "gate_cp_shard_enabled", False):
        return original(layer, hidden_states)
    if layer._gate_cp_chain is None:
        raise RuntimeError("HYV4 CP-sharded gate is not bound to its model")
    return layer._gate_cp_chain.gate(layer._gate_cp_index, hidden_states)


def register_hooks():
    prefix = "sglang.srt.models.hunyuan_v4."
    for name, hook in (
        ("HYV4HCPreLayer.forward", around_ihc_pre),
        ("HYV4HCLayer.post", around_ihc_post),
        ("HYV4HCLayer.post_pre", around_ihc_post_pre),
        ("HYV4HCHeadLayer.forward", around_ihc_head),
        ("HYV4Attention.__init__", around_attention_init),
        ("HYV4Model.__init__", around_model_init),
        ("HYV4Model.forward", around_model_forward),
        ("HYV4Attention.prepare_attention_output_gate", around_prepare_gate),
    ):
        HookRegistry.register(prefix + name, hook, HookType.AROUND)
