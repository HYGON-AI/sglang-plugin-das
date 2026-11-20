from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator as NativeKVCacheConfigurator
from sglang.srt.runtime_context import get_parallel

from hcu_sglang.utils.detection import is_hcu

_scratch_source = ContextVar("hcu_dsa_layer_split_scratch_source", default=None)


def before_draft_alloc(self, *args, **kwargs):
    if not is_hcu():
        return
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool

    target_pool = self.target_worker.model_runner.token_to_kv_pool
    self.draft_runner.dsa_layer_split_scratch_source = None
    if (
        self.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value
        and get_parallel().pp_size == 1
        and self.speculative_algorithm.is_eagle()
        and not self.speculative_algorithm.is_eagle3()
        and self.draft_runner.layer_info.num_effective_layers == 1
        and self.draft_runner.model_config.num_nextn_predict_layers == 1
        and isinstance(target_pool, LayerSplitDSATokenToKVPool)
    ):
        self.draft_runner.dsa_layer_split_scratch_source = target_pool


def around_init_configurator(original, self):
    source = getattr(self, "dsa_layer_split_scratch_source", None)
    token = _scratch_source.set(source)
    try:
        return original(self)
    finally:
        _scratch_source.reset(token)


def after_default_pool_init(result, configurator, kvc):
    from hcu_sglang.layers.glm53_decode import model_matches
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.runtime_context import get_disagg, get_memory

    model_config = kvc.model_config
    hf_config = model_config.hf_config
    parallel = get_parallel()
    if (
        kvc.is_draft_worker
        or not model_matches(hf_config)
        or get_disagg().disaggregation_mode != "prefill"
        or parallel.pp_size != 1
        or parallel.attn_dcp_size != 1
        or not kvc.spec_algorithm.is_eagle()
        or kvc.spec_algorithm.is_eagle3()
        or resolved_view(kvc.server_args).enable_multi_layer_eagle
        or kvc.spec_aux_config.eagle_draft_num_layers != 1
        or model_config.num_nextn_predict_layers != 1
        or get_memory().enable_hisparse
    ):
        return result

    from sglang.srt.layers.cp.utils import (
        get_glm_dsa_cp_layer_shard_info,
        get_layer_shard_range,
    )
    from sglang.srt.utils.common import is_float4_e2m1fn_x2

    rank, shard_size = get_glm_dsa_cp_layer_shard_info(kvc)
    if (
        rank is None
        or shard_size <= 1
        or is_float4_e2m1fn_x2(kvc.kv_cache_dtype)
        or kvc.fused_entry_bytes("full") is not None
    ):
        return result

    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_index_head_dim,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _should_elide_dsa_index_k,
        calculate_mla_kv_cache_dim,
    )
    from hcu_sglang.mem_cache.dsa_index_cache_mode import (
        index_k_cache_bytes_per_token,
        index_k_workspace_bytes_per_token,
        resolve_index_k_cache_mode,
    )

    main_kv_bytes = calculate_mla_kv_cache_dim(
        model_config=model_config, kv_cache_dtype=kvc.kv_cache_dtype,
    ) * kvc.kv_cache_dtype.itemsize
    cache_mode = resolve_index_k_cache_mode(
        kvc.kv_cache_dtype, kvc.pool_page_size, get_dsa_index_head_dim(hf_config),
    )
    index_bytes = index_k_cache_bytes_per_token(cache_mode)
    workspace_bytes = 2 * index_k_workspace_bytes_per_token(cache_mode)
    num_layers = kvc.layer_info.num_effective_layers
    first_layer = kvc.layer_info.start_layer
    elide_index_k = _should_elide_dsa_index_k(is_draft_worker=False)
    rank_costs = []
    for cp_rank in range(shard_size):
        start, end = get_layer_shard_range(cp_rank, shard_size, num_layers)
        draft_owned = int(cp_rank == shard_size - 1)
        owned = end - start + draft_owned
        owned_indexers = draft_owned + sum(
            not elide_index_k or not dsa_layer_skips_topk(hf_config, first_layer + i)
            for i in range(start, end)
        )
        # Main-KV shares one target scratch; Index-K has two independent scratches.
        rank_costs.append(
            (owned + 1) * main_kv_bytes
            + (owned_indexers + 2) * index_bytes
            + workspace_bytes
        )
    configurator._cell_size = math.ceil(max(rank_costs))
    return result


@dataclass(slots=True, kw_only=True)
class HCUKVCacheConfigurator(NativeKVCacheConfigurator):
    dsa_layer_split_scratch_source: Any = field(default_factory=_scratch_source.get)

    def _build_dsa_kv_pool(
        self, *, max_total_num_tokens, max_running_requests, dsa_pool_class,
    ):
        source = self.dsa_layer_split_scratch_source
        if source is None:
            return super(HCUKVCacheConfigurator, self)._build_dsa_kv_pool(
                max_total_num_tokens=max_total_num_tokens,
                max_running_requests=max_running_requests,
                dsa_pool_class=dsa_pool_class,
            )
        from sglang.srt.configs.model_config import (
            dsa_layer_skips_topk, get_dsa_index_head_dim,
            get_dsa_index_kpool, get_dsa_index_kpool_compress,
        )
        from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info
        from sglang.srt.mem_cache.kv_cache_configurator import (
            _should_elide_dsa_index_k, calculate_mla_kv_cache_dim,
        )
        from sglang.srt.runtime_context import get_exec, max_speculative_num_draft_tokens
        from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
        from hcu_sglang.layers.glm53_decode import model_matches

        if model_matches(self.model_config.hf_config):
            rank, size = source.layer_shard_rank, source.layer_shard_size
        else:
            rank, size = get_glm_dsa_cp_layer_shard_info(self)
        if not self.is_draft_worker or rank is None or size <= 1:
            raise ValueError("Shared DSA scratch requires a layer-split draft pool")
        pool_kwargs = {}
        if self.loc_space_scale > 1:
            pool_kwargs["dcp_replicated"] = True
        if _should_elide_dsa_index_k(is_draft_worker=self.is_draft_worker):
            pool_kwargs["skip_topk_layers"] = [
                dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                for layer_id in range(self.layer_info.start_layer, self.layer_info.end_layer)
            ]
        return LayerSplitDSATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config, kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            index_kpool=get_dsa_index_kpool(self.model_config.hf_config),
            index_kpool_compress=get_dsa_index_kpool_compress(self.model_config.hf_config),
            tail_extra_slots=(max_speculative_num_draft_tokens() or 0),
            max_running_requests=max_running_requests,
            layer_shard_rank=rank,
            layer_shard_size=size,
            layer_shard_rank_offset=size - 1,
            layer_split_scratch_source=source,
            **pool_kwargs,
        )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.alloc_memory_pool",
        before_draft_alloc, HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.init_kv_cache_configurator",
        around_init_configurator, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
        after_default_pool_init, HookType.AFTER,
    )
    for target in (
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator",
        "sglang.srt.model_executor.model_runner.KVCacheConfigurator",
    ):
        HookRegistry.register(target, HCUKVCacheConfigurator, HookType.REPLACE)
