# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import __future__
import ast
import inspect
from functools import lru_cache
from types import FunctionType

from hcu_sglang.utils.detection import is_hcu


def supports_decode_radix_mtp(cfg):
    return (
        cfg.disaggregation_mode == "decode"
        and cfg.disaggregation_decode_enable_radix_cache
        and (cfg.speculative_algorithm or "").upper() in ("EAGLE", "NEXTN")
        and cfg.speculative_eagle_topk == 1
        and cfg.dcp_size == 1
        and not cfg.enable_hisparse
        and cfg.disaggregation_transfer_backend != "fake"
    )


@lru_cache(maxsize=1)
def _allow_glm53_mtp(original):
    if (
        original.__module__ != "sglang.srt.arg_groups.pd_disaggregation_hook"
        or original.__name__ != "handle_pd_disaggregation"
        or original.__closure__ is not None
    ):
        raise RuntimeError("Unsupported PD Decode Radix validation function")

    source, first_line = inspect.getsourcelines(original)
    tree = ast.parse("".join(source))
    ast.increment_lineno(tree, first_line - 1)
    expected = ast.dump(
        ast.parse('cfg.speculative_algorithm not in (None, "DSPARK")', mode="eval").body
    )
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.dump(node.test) == expected
    ]
    if len(guards) != 1:
        raise RuntimeError("Official PD Decode Radix speculative validation changed")
    guards[0].test = ast.copy_location(ast.Constant(False), guards[0].test)
    flags = original.__code__.co_flags & __future__.annotations.compiler_flag
    code = compile(tree, original.__code__.co_filename, "exec", flags, dont_inherit=True)
    function_code = next(
        item for item in code.co_consts
        if isinstance(item, type(code)) and item.co_name == original.__name__
    )
    return FunctionType(function_code, original.__globals__, original.__name__)


def resolve_pd_disaggregation(server_args, previous):
    if is_hcu():
        from hcu_sglang.layers.glm53_decode import model_matches
        from sglang.srt.arg_groups.model_override_base import model_config_of
        from sglang.srt.arg_groups.overrides import resolving_view

        if (
            supports_decode_radix_mtp(resolving_view(server_args))
            and model_matches(model_config_of(server_args).hf_config)
        ):
            return _allow_glm53_mtp(previous)(server_args)
    return previous(server_args)


def has_pending_glm53_radix_prompt(req):
    return (
        getattr(req, "glm53_decode_radix_prompt_len", 0)
        and not req.finished()
        and not req.is_retracted
        and req.kv.holds_kv
    )


def after_decode_outputs(normalized_outputs, processor, /, *, batch, **kwargs):
    pending = [req for req in batch.reqs if has_pending_glm53_radix_prompt(req)]
    if not pending:
        return normalized_outputs

    from sglang.srt.mem_cache.radix_cache import RadixKey

    tree_cache = processor.tree_cache
    tree_core = getattr(tree_cache, "tree_core", None)
    is_eagle = getattr(tree_cache, "is_eagle", getattr(tree_core, "is_eagle", False))
    allocator = processor.token_to_kv_pool_allocator
    allocator.free_group_begin()
    try:
        for req in pending:
            prompt_len = req.glm53_decode_radix_prompt_len
            req.glm53_decode_radix_prompt_len = 0
            prompt_key = RadixKey(
                req.full_untruncated_fill_ids[:prompt_len],
                req.extra_key,
                is_bigram=is_eagle,
                cache_salt=req.cache_salt,
            ).page_aligned(tree_cache.page_size)
            if len(prompt_key) > req.kv.cache_protected_len:
                tree_cache.checkpoint(req, up_to=prompt_len)
    finally:
        allocator.free_group_end()
    return normalized_outputs
