from hcu_sglang.utils.detection import is_hcu


def around_resolve_cp(original, server_args, model):
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        resolved_view,
        resolving_view,
    )

    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if not (
        is_hcu()
        and model.hf_config.architectures[0] == "Glm5NextForConditionalGeneration"
        and cfg.enable_prefill_cp
        and view.attn_cp_size > 1
        and cfg.disaggregation_mode == "prefill"
    ):
        return original(server_args, model)
    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    required = {
        "CP must share the entire TP group": view.attn_cp_size == cfg.tp_size,
        "CP strategy must be interleave": cfg.cp_strategy == "interleave",
        "Mooncake transport is required": cfg.disaggregation_transfer_backend == "mooncake",
        "PP size must be one": cfg.pp_size == 1,
        "attention DP size must be one": cfg.attn_dp_size == 1,
        "MoE DP size must be one": cfg.moe_dp_size == 1,
        "DCP size must be one": cfg.dcp_size == 1,
        "mixed chunks are unsupported": not cfg.enable_mixed_chunk,
        "unified memory is unsupported": not cfg.enable_unified_memory,
        "only native MTP with top-k one is supported": (
            cfg.speculative_algorithm is None
            or (
                cfg.speculative_algorithm.upper() in ("EAGLE", "NEXTN")
                and cfg.speculative_draft_model_path in (None, cfg.model_path)
                and cfg.speculative_eagle_topk in (None, 1)
            )
        ),
    }
    for reason, valid in required.items():
        if not valid:
            raise ValueError(f"HCU GLM CP prefill: {reason}")
    linear = mambaish_config(model)
    if getattr(linear, "linear_num_key_heads", None) is not None:
        heads = (linear.linear_num_key_heads, linear.linear_num_value_heads)
    else:
        heads = (linear.linear_attn_config["num_heads"],)
    if any(count % cfg.tp_size for count in heads):
        raise ValueError("GLM linear attention heads must be divisible by TP size")
    declare_resolution(
        server_args,
        "hcu_glm5_cp_pd",
        enable_cp_tp_group_sharing=True,
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )


def after_manager_init(result, manager, *args, **kwargs):
    if not is_hcu():
        return result
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    manager._hcu_glm_layer_split = parallel.enable_dsa_cache_layer_split
    manager._hcu_glm_cp_state = (
        parallel.enable_cp_tp_group_sharing and manager.attn_cp_size > 1
    )
    if manager._hcu_glm_cp_state:
        if not manager.is_hybrid_mla_backend:
            raise ValueError("GLM CP PD requires the hybrid MLA state registration")
        if manager.enable_all_cp_ranks_for_transfer or manager.is_dummy_cp_rank:
            raise ValueError("GLM CP PD requires bootstrap registration as state TP ranks")
        if manager.attn_tp_size != 1 or parallel.tp_size != manager.attn_cp_size:
            raise ValueError("GLM CP state requires CP=TP with attention TP=1")
        if manager.state_tp_size != manager.attn_cp_size:
            raise ValueError("GLM CP PD requires linear state partitioned across CP ranks")
    return result


def around_skip_flags(original, manager, info):
    skip_kv, skip_state = original(manager, info)
    if getattr(manager, "_hcu_glm_cp_state", False):
        if info is None or info.dst_attn_tp_size != 1:
            raise ValueError("GLM CP PD currently requires decode attention TP=1")
        skip_kv = not manager._hcu_glm_layer_split and manager.attn_cp_rank != 0
        skip_state = False
    return skip_kv, skip_state


def around_send_extra(original, manager, req, indices, executor, info=None):
    if not getattr(manager, "_hcu_glm_cp_state", False):
        return original(manager, req, indices, executor, info)
    from hcu_sglang.disagg.protocol import StateType, resolve_state_component_dst_index

    if info is None or info.dst_attn_tp_size != 1:
        raise ValueError("GLM CP PD currently requires decode attention TP=1")
    state_types = manager.kv_args.state_types
    if StateType.MAMBA not in state_types:
        raise ValueError("GLM CP PD is missing its linear attention state")
    mamba_index = state_types.index(StateType.MAMBA)
    dst_index = resolve_state_component_dst_index(
        state_types, info.dst_state_types, mamba_index
    )
    src_dims = manager.kv_args.state_dim_per_tensor[mamba_index]
    dst_dims = info.dst_state_dim_per_tensor[dst_index]
    if not src_dims or not dst_dims:
        raise ValueError("GLM CP PD requires linear state slice metadata")
    if manager.attn_cp_rank != 0:
        retained = {StateType.MAMBA}
        if manager._hcu_glm_layer_split:
            retained.add(StateType.DSA)
        indices = [
            value if state_types[i] in retained else None
            for i, value in enumerate(indices)
        ]
    return original(manager, req, indices, executor, info)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.parallel_hook.resolve_cp_tp_group_sharing",
        around_resolve_cp,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.common.conn.CommonKVManager.__init__",
        after_manager_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._get_dsa_cache_transfer_skip_flags",
        around_skip_flags,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.maybe_send_extra",
        around_send_extra,
        HookType.AROUND,
    )
