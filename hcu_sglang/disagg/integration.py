"""Formal HCU PD registration; optional wire fields preserve legacy positions."""
from __future__ import annotations
from hcu_sglang.utils.detection import is_hcu


def register_hooks():
    if not is_hcu():
        return
    from hcu_sglang.mem_cache import common as cache_common
    cache_common.register_hooks()
    from hcu_sglang.disagg import utils as pd_utils, prefill as pd_prefill
    pd_utils.register_hooks()
    pd_prefill.register_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.disaggregation.base.conn import KVArgs
    from sglang.srt.disaggregation.mooncake import conn as upstream_mooncake
    from . import protocol, common, mooncake, state, worker, validation
    # Mutable runtime KVArgs, distinct from immutable ServerArgs input records.
    if not hasattr(KVArgs, "kv_layout"):
        KVArgs.kv_layout = None
    for target, cls in [
        ("sglang.srt.disaggregation.base.conn.StateType", protocol.StateType),
        ("sglang.srt.disaggregation.common.conn.PrefillServerInfo", protocol.PrefillServerInfo),
        ("sglang.srt.disaggregation.mooncake.conn.TransferInfo", protocol.TransferInfo),
        ("sglang.srt.disaggregation.mooncake.conn.KVArgsRegisterInfo", protocol.KVArgsRegisterInfo),
    ]:
        HookRegistry.register(target, cls, HookType.REPLACE)
    manager = "sglang.srt.disaggregation.common.conn.CommonKVManager."
    for name in ["__init__", "_should_skip_cp_replicated_state_transfer", "register_to_bootstrap",
                 "try_ensure_parallel_info", "_resolve_rank_mapping", "get_mla_kv_ptrs_with_pp"]:
        HookRegistry.register(manager+name, getattr(common,"manager_"+name.strip("_")), HookType.REPLACE)
    receiver = "sglang.srt.disaggregation.common.conn.CommonKVReceiver."
    for name in ["_get_bootstrap_info_from_server", "_connect"]:
        HookRegistry.register(receiver+name, getattr(common,"receiver_"+name.strip("_")), HookType.REPLACE)
    bootstrap = "sglang.srt.disaggregation.common.conn.CommonKVBootstrapServer."
    for name in ["__init__", "_handle_route_put", "_handle_route_get"]:
        HookRegistry.register(bootstrap+name, getattr(common,"bootstrap_"+name.strip("_")), HookType.REPLACE)
    manager = "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager."
    for name in ["_send_kvcache_generic", "send_kvcache", "_is_generic_kvcache_state_type",
                 "_requires_exact_state_index_match", "maybe_send_extra", "_send_slot_state",
                 "_await_transfer_futures", "_send_mamba_state_slice"]:
        HookRegistry.register(manager+name, getattr(mooncake,name.strip("_")), HookType.REPLACE)
    # New subclass overrides have no original attribute to resolve as a target.
    upstream_mooncake.MooncakeKVManager._send_hcu_kda_state = mooncake.send_hcu_kda_state
    upstream_mooncake.MooncakeKVManager._should_skip_cp_replicated_state_transfer = mooncake.should_skip_cp_replicated_state_transfer
    HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVReceiver._register_kv_args",
        mooncake.register_kv_args, HookType.REPLACE)
    from . import transport_common, transport_mooncake
    transport_common.register_hooks()
    transport_mooncake.register_hooks()
    HookRegistry.register(
        manager + "transfer_worker", worker.transfer_worker, HookType.REPLACE
    )
    HookRegistry.register("sglang.srt.disaggregation.utils.setup_state_kv_args",
        state.around_setup_state_kv_args, HookType.AROUND)

    HookRegistry.register(manager+"_validate_peer_state_layout",
        validation.around_validate_peer_state_layout, HookType.AROUND)

    from hcu_sglang.disagg import decode as pd_decode
    pd_decode.register_hooks()
    from hcu_sglang.disagg import pp as pd_pp
    pd_pp.register_hooks()

    from hcu_sglang.disagg import external_kv_state
    external_kv_state.register_hooks()
