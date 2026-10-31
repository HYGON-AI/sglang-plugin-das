"""DeepseekV2AttentionMLA + attention forward mixin hooks."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v2 import methods
from sglang_hcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
    register_mixin_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_v2"
_ATTN = methods.attention

_MLA = (
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla"
)
_MHA = (
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha"
)
_ROCM = (
    "sglang.srt.models.deepseek_common.attention_forward_methods"
    ".forward_mla_fused_rope_rocm"
)
_CPU = (
    "sglang.srt.models.deepseek_common.attention_forward_methods"
    ".forward_mla_fused_rope_cpu"
)

_MLA_METHODS = (
    ("forward_absorb_prepare", "forward_absorb_prepare"),
    ("forward_absorb_core", "forward_absorb_core"),
)

_MHA_METHODS = (
    ("_set_mla_kv_buffer", "_set_mla_kv_buffer"),
    ("_get_mla_kv_buffer", "_get_mla_kv_buffer"),
    (
        "_get_mla_kv_buffer_from_fp8_for_nsa",
        "_get_mla_kv_buffer_from_fp8",
    ),
    ("_concat_and_cast_mha_k", "_concat_and_cast_mha_k"),
    ("_chunked_prefix_attn_mha", "_chunked_prefix_attn_mha"),
    ("forward_normal_chunked_kv_prepare", "forward_normal_chunked_kv_prepare"),
    ("forward_normal_chunked_kv_core", "forward_normal_chunked_kv_core"),
    ("forward_normal_one_shot_prepare", "forward_normal_one_shot_prepare"),
    ("forward_normal_one_shot_core", "forward_normal_one_shot_core"),
)

_ROCM_METHODS = (
    ("forward_absorb_fused_mla_rope_prepare", "forward_absorb_fused_mla_rope_prepare"),
    ("forward_absorb_fused_mla_rope_core", "forward_absorb_fused_mla_rope_core"),
)

_CPU_METHODS = (
    (
        "forward_absorb_fused_mla_rope_cpu_prepare",
        "forward_absorb_fused_mla_rope_cpu_prepare",
    ),
    (
        "forward_absorb_fused_mla_rope_cpu_core",
        "forward_absorb_fused_mla_rope_cpu_core",
    ),
)

_ATTN_CLASS_METHODS = (
    "__init__",
    "prepare_qkv_latent",
)


def register_hooks() -> int:
    n = 0
    n += register_mixin_method_replaces_from_module(
        _MLA, "DeepseekMLAForwardMixin", _MLA_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_module(
        _MHA, "DeepseekMHAForwardMixin", _MHA_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_module(
        _ROCM, "DeepseekMLARocmForwardMixin", _ROCM_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_module(
        _CPU, "DeepseekMLACpuForwardMixin", _CPU_METHODS, _ATTN
    )
    n += register_method_replaces_from_module(
        _BASE,
        "DeepseekV2AttentionMLA",
        _ATTN_CLASS_METHODS,
        _ATTN,
    )
    return n
