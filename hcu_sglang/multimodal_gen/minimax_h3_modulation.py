from contextvars import ContextVar
from dataclasses import dataclass

import torch

from hcu_sglang.multimodal_gen import minimax_h3_quant_env as envs
from hcu_sglang.utils.detection import get_gcn_arch, is_hcu

_ACTIVE = ContextVar("hcu_minimax_h3_fp32_modulation", default=False)
_AFFINE_SCOPE = ContextVar("hcu_minimax_h3_affine_quant", default=None)


@dataclass
class _AffineScope:
    block: object
    projection: object | None = None
    pending: tuple | None = None


def after_layer_init(result, layer, arch, quant_config, *args, **kwargs):
    enabled = False
    if is_hcu() and quant_config is not None:
        from hcu_sglang.multimodal_gen.minimax_h3_quant import MiniMaxH3QuantConfig
        enabled = (
            isinstance(quant_config, MiniMaxH3QuantConfig)
            and arch.adaln_curve_grid is not None
            and quant_config.compressed_adaln
            and quant_config.fused
        )
    layer._hcu_fused_fp32_modulation = enabled
    return result


def around_layer_forward(original, layer, *args, **kwargs):
    active = getattr(layer, "_hcu_fused_fp32_modulation", False)
    scope = None
    if (
        active
        and envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        scope = _AffineScope(layer)
    token = _ACTIVE.set(active)
    scope_token = _AFFINE_SCOPE.set(scope)
    try:
        result = original(layer, *args, **kwargs)
        if scope is not None and scope.pending is not None:
            raise RuntimeError("MiniMax H3 affine input was not consumed by its projection")
        return result
    finally:
        _AFFINE_SCOPE.reset(scope_token)
        _ACTIVE.reset(token)


def _eligible_projection(consumer, projection):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import MiniMaxH3Int8LinearMethod
    from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear

    if (
        type(projection) is MergedColumnParallelLinear
        and projection.tp_size == 1
        and projection.bias is None
        and not projection.gather_output
        and type(projection.quant_method) is MiniMaxH3Int8LinearMethod
        and projection.quant_method.fused
        and not any(
            module._forward_pre_hooks or module._forward_hooks
            for module in (consumer, projection)
        )
    ):
        return projection
    return None


def _affine_projection(block, norm):
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTBlock,
        MiniMaxH3MLP,
    )

    if type(block) is MiniMaxH3DiTBlock and norm is block.norm2:
        if type(block.mlp) is MiniMaxH3MLP:
            return _eligible_projection(block.mlp, block.mlp.fc1)
    return None


def around_rmsnorm_scale_shift(original, x, norm, shift, scale, indices, *, dtype, fused=False):
    scope = _AFFINE_SCOPE.get()
    if scope is None:
        return original(x, norm, shift, scale, indices, dtype=dtype, fused=fused)
    previous = scope.projection
    scope.projection = _affine_projection(scope.block, norm)
    try:
        return original(x, norm, shift, scale, indices, dtype=dtype, fused=fused)
    finally:
        scope.projection = previous


def consume_affine_input(layer, x):
    scope = _AFFINE_SCOPE.get()
    if scope is None or scope.pending is None:
        return None
    projection, normalized, params = scope.pending
    if x is not normalized:
        return None
    if layer is not projection:
        raise RuntimeError("MiniMax H3 affine input reached an unexpected projection")
    scope.pending = None
    return params


def around_scale_shift(original, x, shift, scale, indices, *, dtype):
    if _ACTIVE.get():
        from hcu_sglang.multimodal_gen.minimax_h3_quant import (
            affine,
            supports_fp32_modulation,
            supports_int8_affine_quant,
        )

        if supports_fp32_modulation(x, (shift, scale), indices, dtype=dtype):
            scope = _AFFINE_SCOPE.get()
            if (
                scope is not None
                and scope.projection is not None
                and get_gcn_arch(x.device.index).partition(":")[0] == "gfx936"
                and supports_int8_affine_quant(x, shift, scale, indices)
            ):
                if scope.pending is not None:
                    raise RuntimeError("Previous MiniMax H3 affine input was not consumed")
                scope.pending = (scope.projection, x, (shift, scale, indices))
                return x
            return affine(x, shift, scale, indices)
    return original(x, shift, scale, indices, dtype=dtype)


def around_gate(original, x, gate, other, indices, *, dtype, allow_inplace=True):
    if _ACTIVE.get():
        from hcu_sglang.multimodal_gen.minimax_h3_quant import gate as fused_gate, supports_fp32_modulation
        if supports_fp32_modulation(x, (gate,), indices, dtype=dtype, other=other):
            return fused_gate(x, gate, other, indices)
    return original(x, gate, other, indices, dtype=dtype, allow_inplace=allow_inplace)


def register_hooks(registry=None):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if registry is not None:
        HookRegistry = registry
    prefix = "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
    for name in ("MiniMaxH3DiTBlock", "MiniMaxH3FinalLayer"):
        HookRegistry.register(prefix + name + ".__init__", after_layer_init, HookType.AFTER)
        HookRegistry.register(prefix + name + ".forward", around_layer_forward, HookType.AROUND)
    for name, hook in (
        ("_modulate_rmsnorm_scale_shift", around_rmsnorm_scale_shift),
        ("_modulate_scale_shift", around_scale_shift),
        ("_modulate_gate", around_gate),
    ):
        HookRegistry.register(prefix + name, hook, HookType.AROUND)
