from __future__ import annotations

import json
import math
import os
from contextvars import ContextVar
from functools import cache, wraps
from inspect import unwrap
from types import FunctionType


verify_context = ContextVar("minimax_m31_hisparse_verify", default=None)


def _validate(original, server_args):
    original(server_args)
    from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
    from sglang.srt.configs.model_config import is_minimax_sparse
    from sglang.srt.speculative.ragged_verify import RaggedVerifyMode, read_ragged_verify_mode

    import torch

    cfg = resolving_view(server_args)
    if not cfg.enable_hisparse or not is_minimax_sparse(model_config_of(server_args).hf_config):
        return
    fake = cfg.disaggregation_transfer_backend == "fake" and os.getenv("M31_HISPARSE_FAKE_VALIDATE") == "1"
    if not (
        torch.version.hip
        and os.getenv("HCU_PLUGIN_MINIMAX_M31_FLASHMLA") == "1"
        and cfg.disaggregation_mode == "decode"
        and (cfg.disaggregation_transfer_backend == "mooncake" or fake)
        and cfg.pp_size == cfg.attn_cp_size == cfg.dcp_size == 1
        and not cfg.enable_hierarchical_cache
        and not cfg.disaggregation_decode_enable_radix_cache
        and cfg.disaggregation_decode_host_receive_threshold == 1
        and cfg.speculative_algorithm == "DSPARK"
        and cfg.speculative_num_draft_tokens == 8
        and read_ragged_verify_mode() is RaggedVerifyMode.STATIC
        and cfg.page_size == 128
    ):
        raise ValueError("MiniMax HiSparse requires HIP FlashMLA PD Decode, static DSpark verify8, page size 128 and PP/CP/DCP=1")
    config = json.loads(cfg.hisparse_config or "{}")
    size = config.get("device_buffer_size", 2 * config.get("top_k", 2048))
    ratio = config.get("host_to_device_ratio", 2)
    query = int(os.getenv("M31_HISPARSE_VERIFY_QUERY_CHUNK", "8"))
    compact = fake and os.getenv("M31_HISPARSE_AB_COMPACT_HOT") == "1"
    headwise = os.getenv("M31_HISPARSE_HEADWISE_SWAP") == "1"
    if query not in (4, 8) or (query != 8 and not compact):
        raise ValueError("Split-query MiniMax HiSparse requires fake PD compact staging")
    if headwise and not (
        compact and query == 8 and all(os.getenv(name) == "1" for name in (
            "M31_HISPARSE_STABLE_SLOTS", "M31_HISPARSE_HIP_STABLE_PLAN",
            "M31_HISPARSE_FAST_PLAN", "M31_HISPARSE_DIRECT_VERIFY_INDICES",
        ))
    ):
        raise ValueError("Head-wise MiniMax HiSparse requires fake PD compact staging and direct HIP stable slots")
    min_size = query * 16 * 128 if headwise else (query * 4 * 15 + 2) * 128 if compact else 65536
    if (
        type(size) is not int or size < min_size or size % 128
        or (headwise and size > 256 * 128)
        or type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 1
    ):
        raise ValueError(f"MiniMax HiSparse requires a page-aligned device buffer >= {min_size} and host ratio >= 1")


@cache
def _bind(original, patch):
    from hcu_sglang.models import _minimax_hisparse_runtime as impl
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.mem_cache.pool_host.common import _CUDA_HOST_REGISTERED_RANGES_ATTR

    seed = unwrap(original)
    namespace = {**seed.__globals__, "os": os, "is_hcu": is_hcu}
    namespace["_CUDA_HOST_REGISTERED_RANGES_ATTR"] = _CUDA_HOST_REGISTERED_RANGES_ATTR
    namespace["init_minimax_host_pool"] = FunctionType(impl.init_minimax_host_pool.__code__, namespace)
    patched = FunctionType(patch.__code__, namespace, seed.__name__, seed.__defaults__, seed.__closure__)
    patched.__kwdefaults__ = seed.__kwdefaults__
    return patched


def _around(patch, condition):
    @wraps(patch)
    def hook(original, *args, **kwargs):
        if not condition(*args, **kwargs):
            return original(*args, **kwargs)
        return _bind(original, patch)(*args, **kwargs)
    return hook


def _minimax_pool(pool):
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    return isinstance(pool, MiniMaxSparseKVPool)


def _coordinator_init(self, req_to_token_pool, token_to_kv_pool_allocator, *args, **kwargs):
    return _minimax_pool(token_to_kv_pool_allocator.get_kvcache())


def _minimax_allocator(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
    return _minimax_pool(kvcache)


def _coordinator(self, *args, **kwargs):
    return getattr(self, "m3_flashmla_verify", None) is not None


def _host_pool(self, *args, **kwargs):
    from sglang.srt.mem_cache.pool_host.mha import HiSparseMHATokenToKVPoolHost

    return isinstance(self, HiSparseMHATokenToKVPoolHost) and self.head_num == 4


def _dspark(self, *args, **kwargs):
    return _coordinator(self.model_runner.hisparse_coordinator)


def _minimax_host_mode(*args, **kwargs):
    import torch

    return torch.version.hip is not None


def _round_minimax_size(_result, self, kvc, num_layers):
    if _uses_minimax_hisparse(kvc):
        return math.ceil(_result)


def _uses_minimax_hisparse(kvc):
    from sglang.srt.configs.model_config import is_minimax_sparse
    from sglang.srt.runtime_context import get_memory

    return get_memory().enable_hisparse and is_minimax_sparse(kvc.model_config.hf_config)


def _account_draft_kv(_result, self, kvc):
    if (
        not _uses_minimax_hisparse(kvc)
        or kvc.is_draft_worker
        or not kvc.spec_algorithm.is_dflash_family()
        or kvc.fused_entry_bytes("full") is not None
    ):
        return
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config

    ratio = parse_hisparse_config().host_to_device_ratio
    target = self._compute_cell_size(kvc, kvc.layer_info.num_effective_layers)
    self._cell_size += math.ceil((self._cell_size - target) * (ratio - 1))


@cache
def _hip_runtime(original):
    from hcu_sglang.models import _minimax_hisparse_runtime as impl

    return _bind(original, impl._hip_runtime)()


def _any_minimax_pool(*args, **kwargs):
    return True


def _verify_decode(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    coordinator = self.hisparse_coordinator
    verify = getattr(coordinator, "m3_flashmla_verify", None)
    if verify is None:
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    if not forward_batch.forward_mode.is_target_verify():
        raise RuntimeError("MiniMax HiSparse main attention requires TARGET_VERIFY")
    if verify.compact_hot_buffer and (self.init_blocks, self.local_blocks) != (0, 1):
        raise ValueError("Compact HiSparse requires one forced local page and no forced initial pages")
    token = verify_context.set((verify, layer.layer_id))
    try:
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    finally:
        verify_context.reset(token)


def register_hooks():
    from hcu_sglang.models import _minimax_hisparse_runtime as impl
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def add(target, hook):
        HookRegistry.register("sglang.srt." + target, hook, HookType.AROUND)

    add("arg_groups.hisparse_hook.validate_hisparse", _validate)
    coordinator = "managers.hisparse_coordinator.HiSparseCoordinator."
    add(coordinator + "__init__", _around(impl.coordinator_init, _coordinator_init))
    add(coordinator + "admit_request_direct", _around(impl.admit_request_direct, _coordinator))
    add(coordinator + "alloc_device_buffer", _around(impl.alloc_device_buffer, _coordinator))
    add(coordinator + "request_finished", _around(impl.request_finished, _coordinator))
    pool = "mem_cache.pool_host.mha.MHATokenToKVPoolHost."
    add(pool + "_initialize_transfer_views", _around(impl.host_transfer_views, _host_pool))
    add(pool + "init_kv_buffer", _around(impl.host_init_kv_buffer, _host_pool))
    add("mem_cache.allocator.hisparse.HiSparseTokenToKVPoolAllocator.__init__", _around(impl.allocator_init, _minimax_allocator))
    add("mem_cache.memory_pool.MiniMaxSparseKVPool.__init__", _around(impl.minimax_pool_init, _any_minimax_pool))
    add("speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2._forward_decode", _around(impl.dspark_decode, _dspark))
    add("layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend.forward_decode", _verify_decode)

    host = "mem_cache.pool_host.common."
    add(host + "_cuda_host_register", _around(impl._cuda_host_register, _minimax_host_mode))
    add(host + "_cuda_host_unregister_ranges", _around(impl._cuda_host_unregister_ranges, _minimax_host_mode))
    add(host + "_hip_runtime", _hip_runtime)
    configurator = "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator."
    HookRegistry.register(configurator + "_compute_cell_size", _round_minimax_size, HookType.AFTER)
    HookRegistry.register(configurator + "__init__", _account_draft_kv, HookType.AFTER)
