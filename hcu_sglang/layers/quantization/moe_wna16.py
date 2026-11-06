# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/moe_wna16.py
from __future__ import annotations

import sglang.srt.layers.quantization.moe_wna16 as N
from hcu_sglang.utils.detection import is_hcu

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})
_is_hcu = is_hcu()


def moe_wna16_init(
    self,
    linear_quant_method: str,
    weight_bits: int,
    group_size: int,
    has_zp: bool,
    lm_head_quantized: bool,
    modules_to_not_convert: Optional[List[str]],
    full_config: Dict[str, Any],
) -> None:
    super(N.MoeWNA16Config, self).__init__()
    self.weight_bits = weight_bits
    self.group_size = group_size
    self.has_zp = has_zp
    self.bit8_pack_factor = 8 // self.weight_bits
    self.lm_head_quantized = lm_head_quantized
    self.linear_quant_method = linear_quant_method
    self.full_config = full_config
    self.use_marlin = False
    # Avoid circular import

    if self.linear_quant_method == "gptq":
        self.use_marlin = GPTQMarlinConfig.is_gptq_marlin_compatible(full_config)
    elif self.linear_quant_method == "awq":
        if not _is_hcu:
            capability_tuple = get_device_capability()
            device_capability = (
                -1
                if capability_tuple is None
                else capability_tuple[0] * 10 + capability_tuple[1]
            )
            awq_min_capability = AWQConfig.get_min_capability()
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
    else:
        raise ValueError("moe_wna16 only support gptq and awq.")

    if modules_to_not_convert is None:
        self.modules_to_not_convert = []
    else:
        self.modules_to_not_convert = modules_to_not_convert


def around_moe_wna16_init(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return moe_wna16_init(*args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.moe_wna16.MoeWNA16Config.__init__",
        around_moe_wna16_init,
        HookType.AROUND,
    )
