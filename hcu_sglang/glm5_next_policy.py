from __future__ import annotations

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.runtime_context import get_server_args
from sglang.srt.configs.model_config import get_dsa_index_kpool
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda
from hcu_sglang.utils.detection import is_hcu


def is_glm5_next_hcu(config) -> bool:
    from hcu_sglang.utils.detection import is_hcu

    config = getattr(config, "text_config", config)
    return is_hcu() and getattr(config, "model_type", "") in (
        "glm5_next",
        "glm5v_next",
        "glm5next_text",
        "glm5_next_text",
    )


def get_glm5_next_runtime_args():
    return resolved_view(get_server_args())


def is_glm5_flash_channel_fp8(config, quant_config) -> bool:
    config = getattr(config, "text_config", None) or config
    if (
        not is_hcu()
        or getattr(config, "model_type", None) != "glm5_next_text"
        or getattr(config, "hidden_size", None) != 4096
        or getattr(config, "n_routed_experts", None) != 288
        or get_dsa_index_kpool(config) != 4
        or getattr(config, "qk_rope_head_dim", None) != 0
        or quant_config is None
        or quant_config.get_name() != "compressed_tensors"
    ):
        return False

    from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

    scheme = getattr(quant_config, "target_scheme_map", {}).get("Linear", {})
    weight = scheme.get("weights")
    activation = scheme.get("input_activations")
    return (
        weight is not None
        and activation is not None
        and weight.type == activation.type == QuantizationType.FLOAT
        and weight.num_bits == activation.num_bits == 8
        and weight.strategy == QuantizationStrategy.CHANNEL
        and activation.strategy == QuantizationStrategy.TOKEN
        and not weight.dynamic
        and activation.dynamic
    )


def uses_glm5_flash_fp8_pd_kpool(server_args, hf_config) -> bool:
    config = getattr(hf_config, "text_config", None) or hf_config
    kpool = get_dsa_index_kpool(config)
    if not (
        is_glm5_next_hcu(config)
        and getattr(config, "hidden_size", None) == 4096
        and getattr(config, "n_routed_experts", None) == 288
        and getattr(config, "qk_rope_head_dim", None) == 0
        and kpool == 4
        and server_args.disaggregation_mode in ("prefill", "decode")
        and server_args.dcp_size == 1
        and server_args.kv_cache_dtype == "fp8_e4m3"
    ):
        return False

    quant = getattr(hf_config, "quantization_config", None) or getattr(
        config, "quantization_config", None
    )
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        return False
    groups = quant.get("config_groups") or {}
    group = groups.get("FP8_DYNAMIC") or {}
    weights = group.get("weights") or {}
    activations = group.get("input_activations") or {}
    return (
        len(groups) == 1
        and weights.get("type") == activations.get("type") == "float"
        and weights.get("num_bits") == activations.get("num_bits") == 8
        and weights.get("strategy") == "channel"
        and weights.get("dynamic") is False
        and activations.get("strategy") == "token"
        and activations.get("dynamic") is True
    )


def should_remap_pd_dsa_seed_to_local_slots(
    server_args: "ServerArgs", model_config=None
) -> bool:
    """Whether a PD IndexShare seed can be localized for fused TopK.

    Prefill transfers request-relative positions.  A decode worker can safely
    turn those positions into its own allocator slots only when it owns the
    complete, non-CP page table and does not need a HiSparse page swap.
    """

    hf_config = getattr(model_config, "hf_config", None)
    hcu_glm5_no_kpool = (
        is_glm5_next_hcu(hf_config) and get_dsa_index_kpool(hf_config) <= 1
    )
    return (
        (is_cuda() or is_hcu())
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and server_args.disaggregation_mode == "decode"
        and not server_args.enable_hisparse
        and server_args.attn_cp_size == 1
        and not hcu_glm5_no_kpool
    )
