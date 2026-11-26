# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_sglang.utils.detection import is_hcu


def around_kpool_head_gate(original, *args, **kwargs):
    from sglang.srt.environ import envs

    if envs.SGLANG_HCU_KPOOL_EAGER_HEAD_GATE.get():
        original = getattr(original, "_torchdynamo_orig_callable", original)
    return original(*args, **kwargs)


def after_init_draft_attention_backend(result, worker):
    from sglang.srt.runtime_context import get_spec

    from hcu_sglang.glm5_next_policy import (
        get_glm5_next_runtime_args,
        is_glm5_flash_channel_fp8,
    )
    from hcu_sglang.layers.attention.current_dsa.backend import DeepseekSparseAttnBackend

    backend = worker.draft_extend_attn_backend
    if not isinstance(backend, DeepseekSparseAttnBackend):
        return result
    args = get_glm5_next_runtime_args()
    if (
        args.disaggregation_mode == "null"
        and args.nnodes == args.pp_size == args.attn_cp_size == 1
        and args.enable_dp_attention
        and args.attn_dp_size == args.ep_size == args.tp_size > 1
        and args.speculative_algorithm == "EAGLE"
        and worker.topk == 1
        and getattr(worker, "draft_lm_head_vp", None) is None
        and not get_spec().speculative_use_rejection_sampling
    ):
        target = worker.target_worker.model_runner
        backend.align_nmz_draft_extend_flashmla = is_glm5_flash_channel_fp8(
            target.model_config.hf_config, getattr(target.model, "quant_config", None)
        )
    return result


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.layers.attention.current_dsa import backend, rotation, indexer

    for name in ("DeepseekSparseAttnBackend", "DeepseekSparseAttnMultiStepBackend"):
        HookRegistry.register("sglang.srt.layers.attention.dsa_backend." + name, getattr(backend, name), HookType.REPLACE)

    HookRegistry.register("sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation", rotation.rotate_activation, HookType.REPLACE)

    for target in ("sglang.srt.layers.attention.dsa.dsa_indexer.Indexer", "sglang.srt.models.deepseek_v2.Indexer"):
        HookRegistry.register(target, indexer.Indexer, HookType.REPLACE)
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.init_attention_backend",
        after_init_draft_attention_backend,
        HookType.AFTER,
    )
    for name in ("_get_logits_head_gate", "_project_and_scale_head_gates", "_apply_q_scale_and_softmax_scale"):
        HookRegistry.register(
            f"sglang.srt.layers.attention.dsa.dsa_indexer_kpool.IndexerKPool.{name}",
            around_kpool_head_gate,
            HookType.AROUND,
        )
