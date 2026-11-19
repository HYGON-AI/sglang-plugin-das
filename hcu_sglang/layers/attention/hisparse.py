from functools import lru_cache
from pathlib import Path

from hcu_sglang.utils.detection import is_hcu


def _matches_channel_fp8(config):
    dimensions = {
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "index_topk": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
    }
    if any(getattr(config, key, None) != value for key, value in dimensions.items()):
        return False
    quantization = getattr(config, "quantization_config", None)
    if not isinstance(quantization, dict):
        return False
    groups = quantization.get("config_groups", {})
    if quantization.get("quant_method") != "compressed-tensors" or not groups:
        return False
    for group in groups.values():
        weights = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if not (
            weights.get("type") == activation.get("type") == "float"
            and weights.get("num_bits") == activation.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and weights.get("dynamic") is False
            and weights.get("block_structure") is None
            and weights.get("group_size") is None
            and weights.get("symmetric") is True
            and activation.get("strategy") == "token"
            and activation.get("dynamic") is True
            and activation.get("block_structure") is None
            and activation.get("group_size") is None
            and activation.get("symmetric") is True
        ):
            return False
    return True


def enabled(view=None):
    if not is_hcu():
        return False
    from sglang.srt.arg_groups.model_override_base import model_config_of, resolved_view
    from sglang.srt.runtime_context import get_context
    from hcu_sglang.layers.glm53_decode import model_matches

    if view is None:
        try:
            server_args = get_context().server_args
        except ValueError:
            return False
        if server_args is None:
            return False
        view = resolved_view(server_args)
    config = model_config_of(view).hf_config
    return (
        view.enable_hisparse
        and view.kv_cache_dtype == "fp8_e4m3"
        and model_matches(config)
        and _matches_channel_fp8(config)
    )


def around_backend_resolution(original, view):
    if not enabled(view):
        return original(view)
    if view.speculative_algorithm not in (None, "NONE"):
        raise ValueError(
            "HCU GLM FP8 HiSparse speculative decoding is not aligned with the "
            "installed native reference: the draft residency map is missing."
        )
    if view.dcp_size > 1:
        raise ValueError("HCU GLM FP8 HiSparse does not support DCP.")
    declared = {}
    for field in ("dsa_prefill_backend", "dsa_decode_backend"):
        backend = getattr(view, field)
        if backend is None:
            declared[field] = "tilelang"
        elif backend != "tilelang":
            raise ValueError(
                "HCU GLM FP8 HiSparse requires both DSA backends to be tilelang; "
                "raw and scaled FP8 KV layouts cannot be mixed."
            )
    return declared


def around_validate_backend(original, server_args, attr, label):
    from sglang.srt.arg_groups.model_override_base import resolved_view

    view = resolved_view(server_args)
    if not enabled(view):
        return original(server_args, attr, label)
    if getattr(view, attr) not in (None, "tilelang"):
        raise ValueError(f"HCU GLM FP8 HiSparse requires the tilelang {label} backend.")
    return None


def _source():
    return (
        Path(__file__).resolve().parents[2] / "jit_overlay/csrc/kvcacheio/hisparse.cuh"
    )


@lru_cache(maxsize=None)
def _sparse_module(cache_args):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    template_args = make_cpp_args(*cache_args[1:])
    return load_jit(
        "hcu_hisparse_cache",
        *make_cpp_args(*cache_args),
        cuda_files=[str(_source())],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
        extra_include_paths=[str(_source().parents[2] / "include")],
    )


def around_sparse_module(
    original,
    item_size_bytes,
    block_size,
    num_top_k,
    hot_buffer_size,
    is_mla=False,
    is_dsv4_layout=False,
    top_k_block_size=1,
    top_k_is_blocks=False,
    record_miss_plan=False,
    skip_io=False,
):
    values = (
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    if not enabled():
        return original(*values)
    return _sparse_module(values)


@lru_cache(maxsize=None)
def _copy_module(block_size, is_mla, is_dsv4_layout, skip_io):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = (block_size, is_mla, is_dsv4_layout, skip_io)
    template_args = make_cpp_args(*args)
    return load_jit(
        "hcu_hisparse_copy_planned",
        *make_cpp_args(*args),
        cuda_files=[str(_source())],
        cuda_wrappers=[("copy_cache_planned", f"copy_cache_planned<{template_args}>")],
        extra_include_paths=[str(_source().parents[2] / "include")],
    )


def around_copy_module(original, block_size, is_mla, is_dsv4_layout, skip_io):
    if not enabled():
        return original(block_size, is_mla, is_dsv4_layout, skip_io)
    return _copy_module(block_size, is_mla, is_dsv4_layout, skip_io)


def around_tilelang_sparse_fwd(original, q, kv, indices, sm_scale, d_v=512):
    if not enabled():
        return original(q, kv, indices, sm_scale, d_v)
    import torch
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as kernels

    assert q.ndim == kv.ndim == indices.ndim == 3
    assert d_v == 512 and q.shape[-1] == kv.shape[-1] == 576
    assert indices.shape[-1] == 2048 and kv.dtype == torch.float8_e4m3fn
    if q.dtype != kv.dtype:
        q = q.to(kv.dtype)
    block_i, threads = 64, 256
    block_per_cu, cu = (2, 256) if kernels._is_gfx95_supported else (1, 304)
    ni = indices.shape[-1] // block_i
    inner_iter = kernels._pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
    partial = kernels.sparse_mla_fwd_decode_partial_fp8(
        q.shape[1],
        d_v,
        q.shape[-1] - d_v,
        indices.shape[-1],
        sm_scale=sm_scale,
        block_I=block_i,
        inner_iter=inner_iter,
        threads=threads,
    )
    partial_o, partial_lse = partial(
        q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
    )
    combine = kernels.sparse_mla_fwd_decode_combine(
        q.shape[1],
        d_v,
        ni // inner_iter * block_i,
        head_per_block=4,
        block_I=block_i,
        threads=threads,
    )
    return combine(partial_o, partial_lse)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        (
            "sglang.srt.arg_groups.model_hook._dsa_split_backend_resolution",
            around_backend_resolution,
        ),
        (
            "sglang.srt.arg_groups.hisparse_hook.validate_hisparse_dsa_backend",
            around_validate_backend,
        ),
        (
            "sglang.kernels.ops.kvcache.hisparse._jit_sparse_module",
            around_sparse_module,
        ),
        (
            "sglang.kernels.ops.kvcache.hisparse._jit_copy_planned_module",
            around_copy_module,
        ),
        (
            "sglang.kernels.ops.attention.dsa.tilelang_kernel.tilelang_sparse_fwd",
            around_tilelang_sparse_fwd,
        ),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
