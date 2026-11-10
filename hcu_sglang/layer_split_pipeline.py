# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)

_UPSTREAM_GUARD = ast.parse(
    '''if cfg.enable_dsa_cache_layer_split and cfg.pp_size > 1:
    raise ValueError(
        "--enable-dsa-cache-layer-split is not supported with "
        "pipeline parallelism (pp_size > 1) yet. It requires "
        "prefill context parallelism, and CP + PP has not been "
        "validated for this feature."
    )
'''
).body[0]


def validate_pipeline(view):
    if view.enable_dsa_cache_layer_split and view.hicache_storage_backend is not None:
        raise NotImplementedError(
            "HCU --enable-dsa-cache-layer-split currently supports "
            "HiCache L1/L2 only. --hicache-storage-backend (L3) is "
            "not layer-shard-aware yet."
        )


@lru_cache(maxsize=None)
def adapt_model_adjustments(original):
    from hcu_adapter.lightop_metadata import rewrite_resolution_function

    adapted = rewrite_resolution_function(original, _replace_pipeline_guard)
    logger.info("HCU DSA pipeline validation uses official model resolution, source_sha256=%s",
                adapted.__hcu_source_sha256__)
    return adapted


def _replace_pipeline_guard(tree):
    definition = tree.body[0]
    if definition.name != "handle_model_specific_adjustments":
        raise RuntimeError("Unsupported official model-resolution function")
    expected = ast.dump(_UPSTREAM_GUARD, include_attributes=False)
    matches = [node for node in ast.walk(definition)
               if isinstance(node, ast.If) and ast.dump(node, include_attributes=False) == expected]
    if len(matches) != 1:
        raise RuntimeError(
            "Official DSA pipeline validation changed: expected exactly one inline PP guard, "
            f"found {len(matches)}"
        )
    guard = matches[0]
    replacement = ast.parse(
        "from hcu_sglang.layer_split_pipeline import validate_pipeline as _hcu_validate_pipeline\n"
        "_hcu_validate_pipeline(cfg)\n"
    ).body
    if any(isinstance(node, ast.Name) and node.id == "_hcu_validate_pipeline"
           for node in ast.walk(definition)):
        raise RuntimeError("Official model-resolution local name conflicts with HCU validation")
    for statement in replacement:
        for node in ast.walk(statement):
            ast.copy_location(node, guard)
    for parent in ast.walk(definition):
        for _, values in ast.iter_fields(parent):
            if isinstance(values, list) and guard in values:
                index = values.index(guard)
                values[index:index + 1] = replacement
                break
        else:
            continue
        break
