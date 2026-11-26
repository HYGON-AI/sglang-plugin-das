# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)


def use_native_draft_topk_seed(config) -> bool:
    from sglang.srt.environ import envs

    if not is_hcu() or not envs.SGLANG_HCU_GLM53_FP8_MTP_SEED_FROM_EXTEND.get():
        return False
    from hcu_sglang.layers.rotary import _glm53_fp8_model

    return _glm53_fp8_model(config)


def localize_initial_ifb_seed(forward_batch):
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.layers.attention.dsa.utils import get_disagg
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import get_attn_backend

    spec_info = forward_batch.spec_info
    capture = getattr(spec_info, "dsa_seed_topk_capture", None)
    selected = getattr(spec_info, "dsa_seed_topk_select", None)
    mode = effective_forward_mode(forward_batch)
    if (
        capture is None
        or selected is None
        or mode != ForwardMode.EXTEND
        or not forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
        or get_disagg().disaggregation_mode != "null"
    ):
        return

    seed = capture[: selected.numel()]
    local_slots = get_attn_backend().prepare_mtp_seed_indices(seed, mode)
    if local_slots is not seed:
        seed.copy_(local_slots)
        logger.debug("Localized initial IFB draft seed rows=%s dtype=%s", seed.shape[0], seed.dtype)


def after_init_dsa_index_share_state(result, self):
    config = self.draft_runner.model_config.hf_config
    if not is_hcu() or getattr(config, "model_type", None) != "glm_moe_dsa":
        return result
    if use_native_draft_topk_seed(config):
        logger.info(
            "GLM DSA uses native Draft Extend TopK policy: index_share=%s seed_from_extend=%s",
            self.index_share_for_mtp_iteration,
            self.seed_dsa_topk_from_draft_extend,
        )
        return result

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
