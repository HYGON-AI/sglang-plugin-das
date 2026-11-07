# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single create_moe_runner/apply dispatcher for the W8A8 MoE methods."""

from __future__ import annotations

from typing import Any, Callable


def around_create_moe_runner(
    original: Callable[..., Any], quant_method, layer, moe_runner_config
) -> None:
    from hcu_sglang.layers.quantization import w8a8_aiter, w8a8_lightop

    if w8a8_lightop._requested_lightop():
        return w8a8_lightop.around_create_moe_runner(
            original, quant_method, layer, moe_runner_config
        )
    return w8a8_aiter.around_create_moe_runner(
        original, quant_method, layer, moe_runner_config
    )


def _dispatch_apply(
    lightop_arm: Callable[..., Any],
    aiter_arm: Callable[..., Any],
    original: Callable[..., Any],
    quant_method,
    layer,
    dispatch_output,
):
    from hcu_sglang.layers.quantization import w8a8_lightop

    if w8a8_lightop._uses_lightop_runner(quant_method):
        return lightop_arm(original, quant_method, layer, dispatch_output)
    return aiter_arm(original, quant_method, layer, dispatch_output)


def around_apply_int8(
    original: Callable[..., Any], quant_method, layer, dispatch_output
) -> Any:
    from hcu_sglang.layers.quantization import w8a8_aiter, w8a8_lightop

    return _dispatch_apply(
        w8a8_lightop.around_apply_int8,
        w8a8_aiter.around_apply_int8,
        original,
        quant_method,
        layer,
        dispatch_output,
    )


def around_apply_fp8(
    original: Callable[..., Any], quant_method, layer, dispatch_output
) -> Any:
    from hcu_sglang.layers.quantization import w8a8_aiter, w8a8_lightop

    return _dispatch_apply(
        w8a8_lightop.around_apply_fp8,
        w8a8_aiter.around_apply_fp8,
        original,
        quant_method,
        layer,
        dispatch_output,
    )


def around_compressed_apply_weights(
    original: Callable[..., Any], quant_method, layer, dispatch_output
) -> Any:
    from hcu_sglang.layers.quantization import w8a8_aiter, w8a8_lightop

    return _dispatch_apply(
        w8a8_lightop.around_compressed_apply,
        w8a8_aiter.around_apply_compressed_fp8,
        original,
        quant_method,
        layer,
        dispatch_output,
    )


def around_compressed_process_weights(
    original: Callable[..., Any], quant_method, layer
) -> Any:
    # Not mutually exclusive arms: the deepep hook prepares legacy channel-fp8
    # deepgemm weights and the lightop hook repacks channelwise lightop
    # weights, both after the original loader. Preserve the stacked order the
    # two independent registrations produced (lightop outer, deepep inner).
    from hcu_sglang.layers.moe import deepep_fp8
    from hcu_sglang.layers.quantization import w8a8_lightop

    def chained_original(quant_method_, layer_):
        return deepep_fp8.around_process_weights(original, quant_method_, layer_)

    return w8a8_lightop.around_compressed_process_weights(
        chained_original, quant_method, layer
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for module, class_name in (
        ("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8MoEMethod"),
        ("sglang.srt.layers.quantization.w8a8_fp8", "W8A8FP8MoEMethod"),
    ):
        HookRegistry.register(
            f"{module}.{class_name}.create_moe_runner",
            around_create_moe_runner,
            HookType.AROUND,
        )
        HookRegistry.register(
            f"{module}.{class_name}.apply",
            around_apply_int8 if class_name == "W8A8Int8MoEMethod" else around_apply_fp8,
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8_moe."
        "CompressedTensorsW8A8Fp8MoE.apply_weights",
        around_compressed_apply_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8_moe."
        "CompressedTensorsW8A8Fp8MoE.process_weights_after_loading",
        around_compressed_process_weights,
        HookType.AROUND,
    )
