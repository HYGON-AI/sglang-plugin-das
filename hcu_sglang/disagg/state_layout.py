# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations


def pack_state_types(state_types) -> bytes:
    return ",".join(state_type.value for state_type in (state_types or [])).encode(
        "ascii"
    )


def resolve_state_component_dst_index(src_state_types, dst_state_types, src_index: int):
    if not dst_state_types:
        return src_index
    if not src_state_types:
        raise RuntimeError(
            "Destination state_types are present but source state_types are empty."
        )
    if src_index >= len(src_state_types):
        raise RuntimeError(
            f"Source state component index {src_index} exceeds "
            f"state_types length {len(src_state_types)}."
        )
    state_type = src_state_types[src_index]
    occurrence = sum(
        1 for item in src_state_types[: src_index + 1] if item == state_type
    )
    seen = 0
    for dst_index, dst_state_type in enumerate(dst_state_types):
        if dst_state_type == state_type:
            seen += 1
            if seen == occurrence:
                return dst_index
    raise RuntimeError(
        f"Decode peer is missing state component {state_type!s} "
        f"occurrence {occurrence}."
    )


def get_global_full_attention_layer_ids(manager, start_layer):
    def is_mha_layer_type(layer_type) -> bool:
        layer_type = str(layer_type)
        return layer_type in ("attention", "full_attention", "sliding_attention")

    hf_text_config = getattr(manager.model_config, "hf_text_config", None)
    layer_types = getattr(hf_text_config, "layer_types", None)
    if layer_types is None:
        layer_types = getattr(hf_text_config, "layers_block_type", None)
    if layer_types is None:
        hf_config = getattr(manager.model_config, "hf_config", None)
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types is None:
            layer_types = getattr(hf_config, "layers_block_type", None)
    if layer_types:
        return [
            layer_id
            for layer_id, layer_type in enumerate(layer_types)
            if is_mha_layer_type(layer_type)
        ]
    full_attention_layer_ids = getattr(
        manager.model_config, "full_attention_layer_ids", None
    )
    if not full_attention_layer_ids:
        return None
    full_attention_layer_ids = list(full_attention_layer_ids)
    num_hidden_layers = getattr(manager.model_config, "num_hidden_layers", None)
    if (
        num_hidden_layers is not None
        and len(full_attention_layer_ids) < num_hidden_layers
        and start_layer > 0
        and all(layer_id >= start_layer for layer_id in full_attention_layer_ids)
    ):
        return None
    return full_attention_layer_ids
