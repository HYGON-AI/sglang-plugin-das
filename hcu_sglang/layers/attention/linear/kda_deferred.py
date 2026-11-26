from __future__ import annotations

import torch

from sglang.srt.environ import envs


def initialize_pool(pool):
    if not (
        envs.SGLANG_KDA_REPLAYSSM_DEFER_COMMIT.get()
        and pool.use_hcu_kda_replayssm
    ):
        return
    if not envs.SGLANG_KDA_USE_TILELANG_VERIFY.get():
        raise ValueError("Deferred KDA ReplaySSM requires SGLANG_KDA_USE_TILELANG_VERIFY")
    state = pool.mamba_cache
    slots = state.temporal.shape[1]
    if state.replayssm_rawv.shape[1] != slots:
        raise ValueError("Deferred KDA ReplaySSM requires a shared checkpoint and ring slot space")
    pool.replayssm_pending = torch.zeros(
        (slots,), dtype=torch.int32, device=state.temporal.device
    )
    pool._replayssm_pending_resets = []


def reset_pending(pool):
    resets = pool._replayssm_pending_resets
    if not resets:
        return
    pool._replayssm_pending_resets = []
    stream = torch.cuda.current_stream()
    for slots in resets:
        slots.record_stream(stream)
    pool.replayssm_pending[torch.cat(resets)] = 0


def flush_pending(pool, slots):
    from hcu_sglang.layers.attention.linear.kda_kernels.kda_replayssm_spec_decode import (
        commit_kda_replayssm_spec_all_layers,
    )

    state = pool.mamba_cache
    safe_slots = slots.clamp_min(0)
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=state.temporal,
        rawv_cache=state.replayssm_rawv,
        rawk_cache=state.replayssm_rawk,
        g_cache=state.replayssm_g,
        beta_cache=state.replayssm_beta,
        ssm_state_indices=slots,
        accept_lens=pool.replayssm_pending[safe_slots],
    )
    pool.replayssm_pending[safe_slots] = 0


def around_alloc(original, req_pool, reqs):
    pool = req_pool.mamba_pool
    if getattr(pool, "replayssm_pending", None) is None:
        return original(req_pool, reqs)
    if req_pool.mamba_v2p_table is not None or req_pool.enable_mamba_extra_buffer:
        raise ValueError("Deferred KDA ReplaySSM requires static slots without tracking")
    fresh = [req for req in reqs if not req.kv.holds_mamba]
    result = original(req_pool, reqs)
    if result is not None:
        pool._replayssm_pending_resets.extend(req.kv.mamba_pool_idx.reshape(-1) for req in fresh)
    return result


def around_forward_raw(original, runner, forward_batch, *args, **kwargs):
    req_pool = runner.req_to_token_pool
    pool = getattr(req_pool, "mamba_pool", None)
    if not runner.is_draft_worker and getattr(pool, "replayssm_pending", None) is not None:
        reset_pending(pool)
        mode = forward_batch.forward_mode
        req_indices = forward_batch.req_pool_indices
        if (mode.is_decode() or mode.is_extend_without_speculative()) and req_indices.numel():
            slots = req_pool.translate_mamba_indices(req_pool.get_mamba_indices(req_indices))
            flush_pending(pool, slots)
    return original(runner, forward_batch, *args, **kwargs)


def around_copy(original, pool, src_indices, dst_indices):
    pending = getattr(pool, "replayssm_pending", None)
    if pending is not None:
        reset_pending(pool)
        flush_pending(pool, src_indices)
        pending[dst_indices] = 0
    return original(pool, src_indices, dst_indices)


def around_get_cpu_copy(original, pool, indices):
    if getattr(pool, "replayssm_pending", None) is not None:
        torch.cuda.synchronize(pool.replayssm_pending.device)
        reset_pending(pool)
        flush_pending(pool, indices)
    return original(pool, indices)


def around_load_cpu_copy(original, pool, mamba_cache_cpu, indices):
    pending = getattr(pool, "replayssm_pending", None)
    if pending is not None:
        reset_pending(pool)
        pending[indices] = 0
    return original(pool, mamba_cache_cpu, indices)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        ("sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.alloc", around_alloc),
        ("sglang.srt.mem_cache.memory_pool.MambaPool.copy_from", around_copy),
        ("sglang.srt.mem_cache.memory_pool.MambaPool.get_cpu_copy", around_get_cpu_copy),
        ("sglang.srt.mem_cache.memory_pool.MambaPool.load_cpu_copy", around_load_cpu_copy),
        ("sglang.srt.model_executor.model_runner.ModelRunner._forward_raw", around_forward_raw),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
