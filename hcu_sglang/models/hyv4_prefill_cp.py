# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from functools import lru_cache
from inspect import getsource, signature
from textwrap import dedent
from types import FunctionType

from hcu_sglang.utils.detection import is_hcu

_ARCH = "HYV4ForCausalLM"


def _validate_profile(cfg, hf_config):
    from hcu_sglang.models.hyv4_nextn import supports_mtp_profile

    if not (
        cfg.cp_strategy == "interleave"
        and cfg.tp_size == cfg.attn_cp_size == cfg.ep_size == 8
        and cfg.moe_dense_tp_size == 1
        and cfg.moe_a2a_backend == "deepep"
        and cfg.attention_backend == "dsa"
        and cfg.dsa_prefill_backend == "flashmla_sparse"
        and cfg.dsa_decode_backend == "flashmla_kv"
        and cfg.attn_dp_size == cfg.dp_size == cfg.pp_size == cfg.dcp_size == 1
        and cfg.disaggregation_mode == "null"
        and supports_mtp_profile(cfg)
        and not cfg.enable_mixed_chunk
        and not cfg.enable_dsa_cache_layer_split
        and not cfg.enable_hisparse
        and not cfg.enable_hierarchical_cache
        and not cfg.enable_two_batch_overlap
        and getattr(hf_config, "learnable_sink", False)
    ):
        raise ValueError(
            "HCU HYV4 prefill CP requires interleave CP8/TP8/EP8, "
            "DeepEP, dense TP1, DSA flashmla_sparse prefill and flashmla_kv "
            "decode, without DP, PP, DCP, PD or cache sharding; MTP requires "
            "NEXTN/EAGLE, topk1 and draft_tokens=num_steps+1."
        )
    from hcu_sglang.layers.attention.current_dsa.flashmla_backend import get_flashmla_op

    native = get_flashmla_op("flash_mla_with_kvcache", is_hcu=True)
    if "attn_sink" not in signature(native).parameters:
        raise RuntimeError("HCU HYV4 flashmla_kv requires native attn_sink support")


def _copy_function(fn, **bindings):
    result = FunctionType(
        fn.__code__, dict(fn.__globals__, **bindings),
        fn.__name__, fn.__defaults__, fn.__closure__,
    )
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


@lru_cache(maxsize=1)
def _cp_family_provider(original):
    tree = ast.parse(dedent(getsource(original)))
    definition, = tree.body
    definition.decorator_list = []
    guards = [
        node for node in ast.walk(definition)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cfg.enable_prefill_cp"
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Raise)
        and "--enable-prefill-cp is not supported for HYV4" in ast.unparse(node.body[0])
    ]
    if len(guards) != 1:
        raise RuntimeError("Official HYV4 prefill CP resolution guard has changed")
    guards[0].body = [ast.copy_location(ast.Pass(), guards[0].body[0])]
    scope = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename, "exec"), scope)
    return scope[original.__name__]


@lru_cache(maxsize=2)
def adapt_model_resolution(original):
    from sglang.srt.arg_groups.model_overrides.deepseek_v2 import _deepseek_family_overrides

    bindings = original.__globals__
    collect = bindings["collect_model_override_declarations"]
    invoke = collect.__globals__["_invoke_provider"]

    def invoke_with_cp(fn, args, hf_config):
        if fn is _deepseek_family_overrides:
            fn = _cp_family_provider(fn)
        return invoke(fn, args, hf_config)

    native_collect = _copy_function(collect, _invoke_provider=invoke_with_cp)
    dtype_default = bindings["_dsa_kv_cache_dtype_default"]

    def resolve_kv_cache(view):
        if view.kv_cache_dtype == "auto":
            return {"kv_cache_dtype": "fp8_e4m3"}
        if view.kv_cache_dtype == "fp8_e4m3":
            return {}
        return dtype_default(view)

    def resolve_backends(view):
        from sglang.srt.arg_groups.overrides import _check_dsa_backend_constraints

        _check_dsa_backend_constraints(
            view.kv_cache_dtype,
            view.dsa_prefill_backend,
            view.dsa_decode_backend,
            hip=True,
            dcp_size=view.dcp_size,
        )
        return {}

    return _copy_function(
        original,
        collect_model_override_declarations=native_collect,
        _dsa_kv_cache_dtype_default=resolve_kv_cache,
        _dsa_split_backend_resolution=resolve_backends,
    )


def adapt_if_requested(original, server_args, *, model_arch):
    from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view

    cfg = resolving_view(server_args)
    if not (is_hcu() and model_arch == _ARCH and cfg.enable_prefill_cp):
        return original
    _validate_profile(cfg, model_config_of(server_args).hf_config)
    resolve_model = adapt_model_resolution(original)

    # HYV4 shards at the strategy-based runner boundary. The legacy HCU CP
    # path only splits DSA metadata for aligned batches, leaving HYV4's
    # unsharded model inputs out of sync with attention. Enable the plugin
    # strategy for this launch before any scheduler workers are spawned.
    from sglang.srt.environ import envs

    envs.SGLANG_ENABLE_CP_V2.set(True)
    return resolve_model


def prepare_cp_inputs(model, forward_batch, **kwargs):
    input_embeds = kwargs.pop("input_embeds", None)
    if input_embeds is None:
        input_embeds = model.model.embed_tokens(forward_batch.input_ids)
    return input_embeds, forward_batch.positions, kwargs
