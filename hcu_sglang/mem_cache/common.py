# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import inspect
import logging
from array import array
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, cast

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.srt.environ import envs
from hcu_sglang.mem_cache.cleanup import run_cleanup_step
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_serving, get_spec
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


def suffix_page_indices(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    prefix_len: int,
    seq_len: int,
    page_size: int,
) -> np.ndarray:
    """Pages of a request's tokens [prefix_len, seq_len).

    PD prefill and decode both call this with the same page-aligned prefix
    (decode_prefix_len), so their lists pair up page by page even though their
    rows point at different physical pages.
    """
    return kv_to_page_indices(req_to_token[req_pool_idx, prefix_len:seq_len], page_size)


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    supports_prefix_sharing: bool = True,
    retain_floor: int | None = None,
    component_type: ComponentType = ComponentType.SWA,
    free_segment: Callable[..., None] | None = None,
    eviction_interval: int = 1,
) -> None:
    if not req.kv.holds_kv:
        return

    # For SWA-capable tree caches, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert req.kv.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    # Protected rows limit what can be freed, not where the interval starts.
    evicted_seqlen = req.kv.get_evicted_seqlen(component_type)
    if pre_len - sliding_window_size < evicted_seqlen + eviction_interval:
        return
    dead_lo = (
        req.kv.swa_dead_lo(page_size)
        if component_type == ComponentType.SWA
        else req.kv.cache_protected_len
    )
    evicted_seqlen = max(evicted_seqlen, dead_lo)
    req.kv.set_evicted_seqlen(component_type, evicted_seqlen)

    if not supports_prefix_sharing:
        # Nothing is inserted into a tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Prefix-sharing cache: keep max(window, page) rounded up to whole pages, so
        # a key that ends at page_floor(pre_len) still has a full window of live SWA,
        # and the last leaf, below page_floor(seq_len), is never all-tombstone.
        evict_threshold = pre_len - ceil_align(
            max(sliding_window_size, page_size), page_size
        )
    if retain_floor is not None and supports_prefix_sharing:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Without prefix sharing a retained
        # checkpoint could never be matched, so holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    new_evicted_seqlen = max(evicted_seqlen, evict_threshold)

    if page_size > 1:
        new_evicted_seqlen = (new_evicted_seqlen // page_size) * page_size

    if new_evicted_seqlen > evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, evicted_seqlen:new_evicted_seqlen
        ]
        if free_segment is None:
            assert component_type == ComponentType.SWA
            free_segment = token_to_kv_pool_allocator.free_swa_segment
        free_segment(free_slots, start_pos=evicted_seqlen)
        req.kv.set_evicted_seqlen(component_type, new_evicted_seqlen)


_SWA_TAIL_TAKES_REQ: dict[type, bool] = {}


def _match_prefix_params(**kwargs: Any) -> MatchPrefixParams:
    # Field-filtering dispatch: fork MatchPrefixParams accepts return_full_match;
    # official 0.5.22 declares key/cow_mamba/req only — drop unsupported kwargs
    # instead of crashing match_kv_cache on the official tree
    # ("got an unexpected keyword argument", 2026-11-05 pip-image smoke).
    allowed = {f.name for f in dataclasses.fields(MatchPrefixParams)}
    return MatchPrefixParams(**{k: v for k, v in kwargs.items() if k in allowed})


def _swa_reprefill_tail_tokens(tree_cache: BasePrefixCache, req: Req) -> int:
    # Signature dispatch: the fork tree and this repo's prefix_protocol classes
    # declare swa_reprefill_tail_tokens(self, req=None); official 0.5.22 is
    # parameterless (unified_radix_cache.py:3602). The REPLACE hook runs against
    # both trees, so resolve per class (cached) and call accordingly — a plain
    # (req) call crashes launch_server on the official tree with
    # "takes 1 positional argument but 2 were given" (2026-11-05 pip-image smoke).
    cls = type(tree_cache)
    takes_req = _SWA_TAIL_TAKES_REQ.get(cls)
    if takes_req is None:
        try:
            takes_req = "req" in inspect.signature(
                cls.swa_reprefill_tail_tokens
            ).parameters
        except (AttributeError, ValueError):  # no such method / signature fails
            takes_req = False
        _SWA_TAIL_TAKES_REQ[cls] = takes_req
    if takes_req:
        return tree_cache.swa_reprefill_tail_tokens(req)
    return tree_cache.swa_reprefill_tail_tokens()


def _req_radix_key(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: array[int],
    max_prefix_len: Optional[int] = None,
) -> RadixKey:
    # A reused prefix may carry stale or unwritten SWA (the unified_kv ring, decoder
    # bounded replay); cap the match so the window is re-prefilled.
    reprefill_tail = _swa_reprefill_tail_tokens(tree_cache, req)
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
    if max_prefix_len is not None:
        key_limit = (
            max_prefix_len if key_limit is None else min(key_limit, max_prefix_len)
        )
    return RadixKey(
        token_ids=token_ids,
        extra_key=req.extra_key,
        limit=key_limit,
        cache_salt=req.cache_salt,
    )


def match_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    token_ids: Optional[array] = None,
    *,
    cow_mamba: bool = False,
    max_prefix_len: Optional[int] = None,
    return_full_match: bool = False,
) -> MatchResult:
    """Match token_ids against tree_cache and adopt the hit as req's prefix."""
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    match_result = tree_cache.match_prefix(
        _match_prefix_params(
            key=_req_radix_key(tree_cache, req, token_ids, max_prefix_len),
            return_full_match=return_full_match
            or bool(tree_cache.swa_reprefill_tail_tokens()),
            cow_mamba=cow_mamba,
            req=req,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    req.prefix_len = match_result.device_prefix_len
    (
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(req.prefix_len + req.host_hit_length, max_len)
    req.swa_branching_seqlen = match_result.swa_branching_seqlen
    # A probe match keeps what it did not report; a new round resets both.
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.kv.cache_protected_len = match_result.cache_protected_len
    return match_result


def checkpoint_kv_cache(req: Req, tree_cache: BasePrefixCache) -> None:
    """Publish what the running request has computed so far, unless it is
    barred from the tree."""
    # The tree reads req.finished() to tell a checkpoint from the final
    # insert; a finished request belongs in release_kv_cache.
    assert not req.finished(), f"checkpointing finished request {req.rid}"
    allow_once = getattr(req, "allow_radix_cache_insert_once", False)
    if not req.skip_radix_cache_insert or allow_once:
        req.allow_radix_cache_insert_once = False
        tree_cache.checkpoint(req, up_to=req.extend_end)
    # The next extend resumes after this one, published or not.
    req.prefix_len = req.extend_end


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, *, checkpoint: bool):
    """Give the request's kv row back; with ``checkpoint`` the tree first keeps
    what it can key."""
    if getattr(req, "external_kv_abort_requested", False):
        return _release_aborted_kv_cache(req, tree_cache, checkpoint=checkpoint)
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # A mamba-capable cache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only a mamba-capable tree cache allows freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return
    if tree_cache.claim_kv_row(req):
        # A streaming session detached the kv record to keep the row.
        assert not req.kv.holds_kv
        return

    owned_kv_len = req.owned_kv_len()
    checkpoint = checkpoint and not req.skip_radix_cache_insert
    if checkpoint:
        # A tree that takes over component state (mamba) must see the request
        # finished, or the insert forks the state and the slot leaks.
        assert req.finished() or not tree_cache.supports_mamba(), (
            f"releasing unfinished request {req.rid} into a mamba tree"
        )
        # The fill-id array lags output_ids until the next prepare_for_decode.
        req.refresh_fill_ids()
        tree_cache.checkpoint(req, up_to=owned_kv_len)
    # The protected prefix is not this req's to free.
    tree_cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
    tree_cache.unlock(req.lock)
    req.lock = None
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, checkpointed=checkpoint)

    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    _release_donated_swa_slots(req, tree_cache)
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()


def _release_aborted_kv_cache(
    req: Req, tree_cache: BasePrefixCache, *, checkpoint: bool
):
    """Resume abort cleanup using main's ownership API, never repeat opaque frees."""

    def snapshot():
        if not req.kv.holds_kv:
            return None
        return (req.owned_kv_len(), req.kv.kv_allocated_len, req.kv.cache_protected_len)

    layout = run_cleanup_step(req, "kv.layout", snapshot)
    if layout is None:
        if req.kv.holds_mamba:
            run_cleanup_step(
                req,
                "kv.early_mamba",
                lambda: tree_cache.req_to_token_pool.mamba_allocator.free(
                    req.kv.mamba_pool_idx.unsqueeze(-1)
                ),
            )
            req.kv.mamba_pool_idx = None
        return
    owned, allocated, protected = layout
    if run_cleanup_step(req, "kv.claim", lambda: tree_cache.claim_kv_row(req)):
        return
    checkpoint = checkpoint and not req.skip_radix_cache_insert
    if checkpoint:
        req.refresh_fill_ids()
        run_cleanup_step(
            req, "kv.checkpoint", lambda: tree_cache.checkpoint(req, up_to=owned)
        )
    run_cleanup_step(
        req, "kv.row", lambda: tree_cache.free_kv_row(req.kv, [(protected, owned)])
    )
    run_cleanup_step(req, "kv.unlock", lambda: tree_cache.unlock(req.lock))
    req.lock = None
    run_cleanup_step(
        req,
        "kv.overallocated",
        lambda: _release_overallocated_kv_indices(req, owned, allocated, tree_cache),
    )
    run_cleanup_step(
        req,
        "kv.on_release",
        lambda: tree_cache.on_release(req, checkpointed=checkpoint),
    )
    run_cleanup_step(
        req, "kv.donated_swa", lambda: _release_donated_swa_slots(req, tree_cache)
    )
    run_cleanup_step(
        req, "kv.request_slot", lambda: tree_cache.req_to_token_pool.free(req)
    )
    req.kv.mark_kv_released()


def _release_donated_swa_slots(req: Req, tree_cache: BasePrefixCache) -> None:
    """Free SWA left ownerless by a DSV4 decode-radix prompt donation.

    The donated radix leaf is full-only: it keeps the full pages and tombstones
    their SWA, and the request's own release treats that prefix as tree-owned.
    Nobody frees the SWA underneath, so return it here, after the last decode
    step. swa_evicted_seqlen marks what the window-aware path already took.
    """
    donated = int(getattr(req, "dsv4_donated_swa_len", 0) or 0)
    if donated <= 0 or not req.kv.holds_kv:
        return
    # Clear first: this must run once even if the free below is a no-op.
    req.dsv4_donated_swa_len = 0
    allocator = tree_cache.token_to_kv_pool_allocator
    if not hasattr(allocator, "free_swa"):
        return
    # Deliberately not bounded by swa_evicted_seqlen: free_swa_out_of_window_slots
    # raises that watermark to cache_protected_len without freeing anything, so
    # for the donated prefix it reads as evicted while the slots are still held.
    indices = tree_cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :donated]
    allocator.free_swa(indices)


from sglang.srt.mem_cache import common as _NATIVE
from hcu_sglang.utils.detection import is_hcu as _is_hcu_device

RetractionBackup = _NATIVE.RetractionBackup
kv_to_page_indices = _NATIVE.kv_to_page_indices
kv_to_page_num = _NATIVE.kv_to_page_num
page_align_floor = _NATIVE.page_align_floor
coalesce_ranges = _NATIVE.coalesce_ranges
free_kv_row_segments = _NATIVE.free_kv_row_segments
touch_waiting_prefix = _NATIVE.touch_waiting_prefix
evict_from_tree_cache = _NATIVE.evict_from_tree_cache
_evict_until_allocatable = _NATIVE._evict_until_allocatable
dsv4_needs_rebootstrap = _NATIVE.dsv4_needs_rebootstrap
backup_kv_cache = _NATIVE.backup_kv_cache
restore_kv_cache = _NATIVE.restore_kv_cache
discard_kv_cache_backup = _NATIVE.discard_kv_cache_backup
_release_overallocated_kv_indices = _NATIVE._release_overallocated_kv_indices
available_and_evictable_str = _NATIVE.available_and_evictable_str


def register_hooks():
    if not _is_hcu_device():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    _NATIVE.suffix_page_indices = suffix_page_indices
    HookRegistry.register("sglang.srt.mem_cache.common.free_swa_out_of_window_slots", free_swa_out_of_window_slots, HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.common._req_radix_key", _req_radix_key, HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.common.match_kv_cache", match_kv_cache, HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.common.checkpoint_kv_cache", checkpoint_kv_cache, HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.common.release_kv_cache", release_kv_cache, HookType.REPLACE)
    _NATIVE._release_aborted_kv_cache = _release_aborted_kv_cache
    _NATIVE._release_donated_swa_slots = _release_donated_swa_slots
