# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)


def after_init_dsa_index_share_state(result, self):
    config = self.draft_runner.model_config.hf_config
    if is_hcu() and getattr(config, "model_type", None) == "glm_moe_dsa":
        self.seed_dsa_topk_from_draft_extend = False
        logger.info(
            "GLM DSA recomputes first draft TopK: index_share=%s seed_from_extend=%s",
            self.index_share_for_mtp_iteration,
            self.seed_dsa_topk_from_draft_extend,
        )
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # anchor-baseline(②): EagleDraftWorker._init_dsa_index_share_state — SGLang 0.5.22/baa9bef: self-only initializer with sharing, width and seed fields.
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._init_dsa_index_share_state",
        after_init_dsa_index_share_state,
        HookType.AFTER,
    )
    return 1
