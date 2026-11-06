# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sglang.srt.layers.quantization.auto_round as N
from hcu_sglang.utils.detection import is_hcu

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})


def get_layer_config(self, layer, layer_name: str):
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    def get_config(name: str, quantized: bool = True):
        if not self.extra_config:
            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # Exact match first
        if name in self.extra_config:
            cfg = self.extra_config[name]
            return (
                cfg.get("bits", self.weight_bits if quantized else 16),
                cfg.get("group_size", self.group_size if quantized else -1),
                cfg.get("sym", self.sym if quantized else True),
            )

        REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
        for pattern, cfg in self.extra_config.items():
            if not isinstance(pattern, str) or not any(
                c in REGEX_SPECIAL_CHARS for c in pattern
            ):
                continue

            try:
                if re.fullmatch(pattern, name):
                    return (
                        cfg.get("bits", self.weight_bits if quantized else 16),
                        cfg.get("group_size", self.group_size if quantized else -1),
                        cfg.get("sym", self.sym if quantized else True),
                    )
            except re.error:
                # Invalid regex, ignore.
                continue

        return (
            self.weight_bits if quantized else 16,
            self.group_size if quantized else -1,
            self.sym if quantized else True,
        )

    # 1. Exact match from config
    if self.extra_config and layer_name in self.extra_config:
        return get_config(layer_name)

    # 2. Determine whether layer should be quantized
    quantized = not isinstance(layer, ParallelLMHead)
    if self.block_name_to_quantize:
        quantized = any(
            layer_name.startswith(name) for name in self.block_name_to_quantize
        )

    # 3. Handle fused MoE
    if self.extra_config and (
        "fusedmoe" in layer.__class__.__name__.lower()
        or isinstance(layer, FusedMoE)
    ):
        moe_configs = [
            get_config(name, quantized)
            for name in self.extra_config
            if name.startswith(layer_name)
        ]
        if moe_configs:
            if len(set(moe_configs)) == 1:
                return moe_configs[0]
            raise ValueError(
                f"Fused MoE layer '{layer_name}' requires "
                f"consistent quant config for all sub-layers"
            )

    # 4. Handle fused QKV or other patterns
    if self.extra_config:
        for fusion_key, sub_keys in self.packed_modules_mapping.items():
            if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                sub_names = [
                    layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                ]
                sub_configs = [get_config(name, quantized) for name in sub_names]
                if len(set(sub_configs)) == 1:
                    return sub_configs[0]
                raise ValueError(
                    f"Fused module '{layer_name}' requires "
                    f"consistent quant config for {sub_names}"
                )

    # 5. Fallback or try a regular expression match
    return get_config(layer_name, quantized)


def around_get_layer_config(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return get_layer_config(*args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.auto_round.AutoRoundConfig.get_layer_config",
        around_get_layer_config,
        HookType.AROUND,
    )
