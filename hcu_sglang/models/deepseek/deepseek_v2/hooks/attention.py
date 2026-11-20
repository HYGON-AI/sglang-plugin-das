"""DeepseekV2AttentionMLA + attention forward mixin hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.hook_utils import (
    register_method_replaces_from_path,
    register_mixin_method_replaces_from_path,
)

_BASE = "sglang.srt.models.deepseek_v2"
_ATTN = "hcu_sglang.models.deepseek.deepseek_v2.methods.attention"

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
        "_get_mla_kv_buffer_from_fp8_for_dsa",
        "_get_mla_kv_buffer_from_fp8_for_dsa",
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


def around_forward_core(original, self, intermediate_state):
    from sglang.srt.environ import envs

    if not envs.SGLANG_HCU_GLM53_FP8_TOKEN_INVERSE.get():
        return original(self, intermediate_state)
    from hcu_sglang.layers.quantization.glm53_fp8_token_inverse import attention_scope

    batch = method = None
    if isinstance(intermediate_state, tuple) and len(intermediate_state) == 4 and intermediate_state[3] is not None:
        method, batch = intermediate_state[1:3]
    with attention_scope(self, batch, method):
        return original(self, intermediate_state)


def register_hooks() -> int:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    n = 0
    n += register_mixin_method_replaces_from_path(
        _MLA, "DeepseekMLAForwardMixin", _MLA_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_path(
        _MHA, "DeepseekMHAForwardMixin", _MHA_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_path(
        _ROCM, "DeepseekMLAFusedRopeRocmForwardMixin", _ROCM_METHODS, _ATTN
    )
    n += register_mixin_method_replaces_from_path(
        _CPU, "DeepseekMLACpuForwardMixin", _CPU_METHODS, _ATTN
    )
    n += register_method_replaces_from_path(
        _BASE,
        "DeepseekV2AttentionMLA",
        _ATTN_CLASS_METHODS,
        _ATTN,
    )
    HookRegistry.register(
        f"{_BASE}.DeepseekV2AttentionMLA.forward_core",
        around_forward_core,
        HookType.AROUND,
    )
    n += 1
    return n
