# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_sglang.utils.detection import is_hcu


def _is_channel_w4a8(scheme):
    if scheme is None:
        return False
    weight = scheme.get("weights")
    activation = scheme.get("input_activations")
    return (
        weight is not None
        and activation is not None
        and weight.type == "int"
        and activation.type == "int"
        and weight.num_bits == 4
        and activation.num_bits == 8
        and weight.strategy == "channel"
        and activation.strategy == "token"
        and not weight.dynamic
        and activation.dynamic
        and weight.symmetric
        and activation.symmetric
        and weight.actorder is None
    )


def _is_dsv41_config(scheme_map, ignore):
    return any("engram" in pattern for pattern in ignore) and any(
        "w[123]" in target and _is_channel_w4a8(scheme)
        for target, scheme in scheme_map.items()
    )


def around_init(original, instance, *args, **kwargs):
    schemes = kwargs.get("target_scheme_map", args[0] if args else {})
    ignore = kwargs.get("ignore", args[1] if len(args) > 1 else ())
    saved = {}
    if is_hcu() and _is_dsv41_config(schemes, ignore):
        saved = {
            target: scheme["input_activations"]
            for target, scheme in schemes.items()
            if _is_channel_w4a8(scheme)
        }
    result = original(instance, *args, **kwargs)
    for target, activation in saved.items():
        instance.target_scheme_map[target]["input_activations"] = activation
    return result


def _projection_scheme(config, layer, prefix):
    config._add_fused_moe_to_target_scheme_map()
    try:
        schemes = [
            config.get_scheme_dict(layer, prefix + name)
            for name in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
    except ValueError:
        if not _is_dsv41_config(config.target_scheme_map, config.ignore):
            raise
        schemes = [
            config.get_scheme_dict(layer, prefix + name)
            for name in (".0.w1", ".0.w3", ".0.w2")
        ]
    if not all(scheme == schemes[0] for scheme in schemes):
        raise ValueError("All MoE projections need to have same quantization scheme but found multiple")
    return schemes[0]


def around_get_quant_method(original, config, layer, prefix):
    if not is_hcu() or config.quant_format != "pack-quantized":
        return original(config, layer, prefix)
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

    if not isinstance(layer, FusedMoE):
        return original(config, layer, prefix)
    a2a = get_moe_a2a_backend()
    if not (a2a.is_megamoe() or (a2a.is_deepep() and get_moe_runner_backend().is_deep_gemm())):
        return original(config, layer, prefix)
    if not _is_channel_w4a8(_projection_scheme(config, layer, prefix)):
        return original(config, layer, prefix)
    from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import HCUW4A8Int8DeepEPMoEMethod

    return HCUW4A8Int8DeepEPMoEMethod(config, use_mega_moe=a2a.is_megamoe())


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    owner = "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.CompressedTensorsConfig"
    HookRegistry.register(owner + ".__init__", around_init, HookType.AROUND)
    HookRegistry.register(owner + ".get_quant_method", around_get_quant_method, HookType.AROUND)
