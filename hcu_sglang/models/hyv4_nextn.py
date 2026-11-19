# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import MethodType

from hcu_sglang.models.hyv4_mtp_metadata import (
    normalize_hyv4_mtp_weight_name,
    validate_hyv4_mtp_metadata,
)
from hcu_sglang.utils.detection import is_hcu


def supports_mtp_profile(cfg):
    if cfg.speculative_algorithm is None:
        return True
    steps = getattr(cfg, "speculative_num_steps", None)
    drafts = getattr(cfg, "speculative_num_draft_tokens", None)
    return (
        cfg.speculative_algorithm in ("NEXTN", "EAGLE")
        and getattr(cfg, "speculative_eagle_topk", None) == 1
        and isinstance(steps, int) and steps > 0
        and isinstance(drafts, int) and drafts == steps + 1
    )


def normalize_mtp_quant_config(quant_config):
    if quant_config is None:
        return None
    result = copy.deepcopy(quant_config)

    def normalize(name):
        for prefix in ("model.mtp.layers.0", "model.mtp_layers.0",
                       "mtp.layers.0", "mtp_layers.0"):
            for special in ("enorm", "hnorm", "eh_proj", "shared_head.norm"):
                name = name.replace(prefix + "." + special, "model." + special)
            name = name.replace(prefix, "model.decoder")
        return name.replace(".self_attn.g_proj", ".self_attn.linear_gate")

    for field in ("ignored_layers", "ignore", "linear_targets"):
        values = getattr(result, field, None)
        if values is not None:
            setattr(result, field, list(dict.fromkeys(normalize(x) for x in values)))
    if hasattr(result, "linear_fp8_config"):
        result.linear_fp8_config = None
    return result


def around_mtp_quant_config(original, quant_config):
    if not is_hcu():
        return original(quant_config)
    return normalize_mtp_quant_config(quant_config)


def around_nextn_init(original, model, config, quant_config=None, prefix=""):
    if not is_hcu():
        return original(model, config, quant_config=quant_config, prefix=prefix)
    result = original(
        model, config, quant_config=normalize_mtp_quant_config(quant_config), prefix=prefix
    )
    from hcu_sglang.models.hunyuan_v4 import _prefill_cp_enabled
    from hcu_sglang.models.hyv4_prefill_cp import prepare_cp_inputs

    if _prefill_cp_enabled():
        model.prepare_cp_inputs = MethodType(prepare_cp_inputs, model)
    return result


def around_nextn_model_init(original, model, *args, **kwargs):
    result = original(model, *args, **kwargs)
    if not is_hcu():
        return result
    mlp = model.decoder.mlp
    shared = getattr(mlp, "shared_experts", None)
    if shared is not None:
        shared.swiglu_limit = None
        if hasattr(shared, "act_fn"):
            shared.act_fn.limit = None
    from sglang.srt.environ import envs

    if envs.SGLANG_HY4_LINEAR_GATE_CP_SHARD.get():
        from hcu_sglang.models.hyv4_gate_cp_shard import build_gate_cp_chain
        model.gate_cp_chain = build_gate_cp_chain([model.decoder.self_attn])
    return result


def around_nextn_model_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    if not is_hcu():
        return original(
            model, input_ids, positions, forward_batch, input_embeds=input_embeds
        )
    from hcu_sglang.models.hyv4_env_hooks import around_model_forward

    result = around_model_forward(
        original, model, input_ids, positions, forward_batch, input_embeds=input_embeds
    )
    if model.alt_stream is not None:
        import torch
        torch.cuda.current_stream().wait_stream(model.alt_stream)
    return result


def around_nextn_load_weights(original, model, weights):
    quant = getattr(model, "quant_config", None)
    if not (
        is_hcu()
        and getattr(quant, "checkpoint_format", None) == "hy4_compressed_tensors_w4a8"
    ):
        return original(model, weights)
    from sglang.srt.models.hunyuan_v4 import permute_hyv4_indexer_weight
    from hcu_sglang.models.hunyuan_v4 import adapt_hyv4_packed_weight

    prefix = model._initialize_nextn_conf(True).nextn_layer_prefix
    suffix = "_inv" if any(
        name.endswith(".weight_scale_inv") for name, _ in model.named_parameters()
    ) else ""
    metadata = {}
    uses_ct_loader = getattr(quant, "uses_hyv4_deepep", lambda: False)()
    seen = set()

    def mapped_weights():
        for name, weight in weights:
            relative = normalize_hyv4_mtp_weight_name(name)
            if relative is None:
                continue
            if relative in seen:
                raise ValueError(f"Duplicate HYV4 MTP weight: {relative}")
            seen.add(relative)
            if uses_ct_loader:
                metadata_name, metadata_weight = adapt_hyv4_packed_weight(name, weight, quant)
            else:
                name, weight = adapt_hyv4_packed_weight(name, weight, quant)
                if name is None:
                    continue
                relative = normalize_hyv4_mtp_weight_name(name)
                metadata_name, metadata_weight = name, weight
            if metadata_name is not None:
                metadata_key = normalize_hyv4_mtp_weight_name(metadata_name)
                if metadata_key in metadata:
                    raise ValueError(f"Duplicate HYV4 MTP weight: {metadata_key}")
                metadata[metadata_key] = {
                    "shape": list(metadata_weight.shape), "dtype": str(metadata_weight.dtype)
                }
            name = prefix + "." + relative
            weight = permute_hyv4_indexer_weight(name, weight, model.config)
            if name.endswith(".weight_scale"):
                name += suffix
            yield name, weight
        validate_hyv4_mtp_metadata(
            vars(model.config), metadata, packed_experts=True, mixed_int8=True
        )

    return model.do_load_weights(mapped_weights(), is_nextn=True)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    prefix = "sglang.srt.models.hunyuan_v4_nextn."
    for name, hook in (
        ("_mtp_quant_config", around_mtp_quant_config),
        ("HYV4ForCausalLMNextN.__init__", around_nextn_init),
        ("HYV4ForCausalLMNextN.load_weights", around_nextn_load_weights),
        ("HYV4ModelNextN.__init__", around_nextn_model_init),
        ("HYV4ModelNextN.forward", around_nextn_model_forward),
    ):
        HookRegistry.register(prefix + name, hook, HookType.AROUND)
