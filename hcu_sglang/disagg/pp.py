# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from hcu_sglang.disagg.hidden_state import (
    get_pd_hidden_capture_layer_ids,
    get_pd_hidden_req_state as pd_hidden_state,
)
from hcu_sglang.disagg.utils import DisaggregationMode
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import DynamicGradMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

_is_hcu = is_hcu()
PPTransferStatus = Tuple[List[str], List[str]]


def _pp_ordered_union(left: List[str], right: List[str]) -> List[str]:
    seen = set(left)
    merged = list(left)
    for rid in right:
        if rid not in seen:
            seen.add(rid)
            merged.append(rid)
    return merged


def _pp_ordered_intersection(left: List[str], right: List[str]) -> List[str]:
    right_set = set(right)
    return [rid for rid in left if rid in right_set]


def _pp_merge_transfer_status(
    previous: PPTransferStatus,
    current: PPTransferStatus,
) -> PPTransferStatus:
    previous_success, previous_failed = previous
    current_success, current_failed = current
    failed = _pp_ordered_union(previous_failed, current_failed)
    success = _pp_ordered_intersection(previous_success, current_success)
    if failed:
        failed_set = set(failed)
        success = [rid for rid in success if rid not in failed_set]
    return success, failed


@DynamicGradMode()
def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        pop bootstrapped reqs (PPConsensusStore + pp_sync polls)
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        Release Requests still use the two-pass PP consensus. Bootstrap uses
        PPConsensusStore instead of the legacy two-round RID sync.

        """
        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size if _is_hcu else None
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: PPTransferStatus = ([], [])
        release_rids: Optional[PPTransferStatus] = None
        send_bootstrapped_work = []
        send_consensus_bootstrapped_work = []
        send_transfer_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (
                    mb_id + get_parallel().pp_size
                ) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.ingest_requests()

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                if _is_hcu:
                    bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                    bmbs[mb_id] = bootstrapped_rids
                    self._pp_commit_comm_work(send_bootstrapped_work)
                else:
                    self.process_bootstrapped_queue()

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk(
                    last_batch=self.last_batch, running_batch=self.running_batch
                )
                self._process_hicache_events()
                prefill_plan = self.get_new_batch_prefill(self.running_batch)
                batch = prefill_plan.batch_to_run
                self.running_batch = prefill_plan.running_batch
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()

                if get_parallel().pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_proxy_send_work()
                if cur_batch:
                    if self.enable_staging:
                        self.maybe_prefetch_staging_for_batch(cur_batch)
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )
                if get_parallel().pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if _is_hcu:
                    send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                        self._pp_pd_send_consensus_bootstrapped_ids(
                            bmbs,
                            next_first_rank_mb_id,
                            consensus_bootstrapped_rids,
                            bootstrapped_rids,
                        )
                    )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if _is_hcu:
                    if bmbs[next_mb_id] is not None:
                        next_consensus_bootstrapped_rids = (
                            self._pp_recv_pyobj_from_prev_stage()
                        )
                        next_consensus_bootstrapped_rids = (
                            self.process_bootstrapped_queue(
                                next_consensus_bootstrapped_rids
                            )
                        )
                    self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    if _is_hcu:
                        send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                            bootstrapped_rids, async_send=True
                        )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch:
                        self._pp_send_proxy_to_next_stage(result)

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self._sched_idled = True
                if len(self.disagg_prefill_inflight_queue) == 0:
                    self.on_idle()


def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]] = None
    ):
        if not _is_hcu:
            reqs = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            self.waiting_queue.extend(reqs)
            return
        # Finished consensus bootstrap requests.
        # In PP disagg prefill parallel mode, requests may already be enqueued into
        # waiting_queue before bootstrap notify is done. Here we only advance
        # bootstrap/transfer states.
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    pp_good_rids=good_consensus_bootstrapped_rids,
                    pp_bad_rids=bad_consensus_bootstrapped_rids,
                )
            )

            # if ready_reqs:
            #     self._try_send_prefill_kv_ready_batch(ready_reqs)
            self.waiting_queue.extend(good_reqs)
            return [
                [req.rid for req in good_reqs],
                _pp_ordered_union(
                    bad_consensus_bootstrapped_rids,
                    [req.rid for req in failed_reqs],
                ),
            ]
        return None


def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # Probe local credits without reserving resources before consensus.
            good_bootstrapped_rids, bad_bootstrapped_rids = (
                self.disagg_prefill_bootstrap_queue.get_ready_bootstrapped_rids_for_pp()
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = (
                self.disagg_prefill_bootstrap_queue.get_ready_bootstrapped_rids_for_pp()
            )
            good_bootstrapped_rids = self._pp_ordered_intersection(
                prev_good_bootstrapped_rids, curr_good_bootstrapped_rids
            )
            bad_bootstrapped_rids = self._pp_ordered_union(
                prev_bad_bootstrapped_rids, curr_bad_bootstrapped_rids
            )
        # Route locally-aborted reqs through the bad-union consensus so every PP
        # rank flushes them in the same consensus round, regardless of when the
        # AbortReq reaches each rank and regardless of whether
        # disagg_kv_sender.abort() drives the poll to Failed (it is optional).
        aborted_rids = {
            req.rid
            for req in self.disagg_prefill_bootstrap_queue.queue
            if isinstance(req.finished_reason, FINISH_ABORT)
        }
        good_bootstrapped_rids, bad_bootstrapped_rids = self._route_aborts_to_bad(
            good_bootstrapped_rids, bad_bootstrapped_rids, aborted_rids
        )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]


def _pp_pd_get_prefill_transferred_ids(
        self: Scheduler,
    ) -> PPTransferStatus:
        # get the current stage transfer success
        current_status = self.get_transferred_rids()
        if self.pp_group.is_first_rank:
            transferred_rids = current_status
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            previous_status = self._pp_recv_pyobj_from_prev_stage()
            transferred_rids = _pp_merge_transfer_status(
                previous=previous_status,
                current=current_status,
            )
        return transferred_rids


def _pp_should_owner_direct_pd_hidden(
        self: Scheduler, batch: ScheduleBatch
    ) -> bool:
        if batch and batch.spec_algorithm.is_dspark():
            return False
        if not hasattr(self, "disagg_prefill_bootstrap_queue"):
            return False
        if not batch or not get_pd_hidden_capture_layer_ids(batch.reqs):
            return False
        capture_reqs = [
            req for req in batch.reqs if pd_hidden_state(req).capture_layer_ids
        ]
        if not capture_reqs:
            return False
        if any(req.pending_bootstrap for req in capture_reqs):
            return False
        return all(
            bool((pd_hidden_state(req).meta or {}).get("streaming_hidden", False))
            for req in capture_reqs
        )


def _pp_maybe_send_dspark_owner_direct_hidden(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not self._pp_should_owner_direct_pd_hidden(batch):
            return
        send_owner_direct = getattr(
            self, "send_dspark_owner_direct_hidden_for_batch", None
        )
        if send_owner_direct is None:
            return
        if send_owner_direct(batch, result):
            self._pp_strip_pd_aux_hidden_from_proxy(result)


def _pp_strip_pd_aux_hidden_from_proxy(
        self: Scheduler, result: GenerationBatchResult
    ) -> None:
        proxy = result.pp_hidden_states_proxy_tensors
        if proxy is None:
            return
        tensors = proxy.tensors
        aux_keys = [key for key in tensors if key.startswith("pd_aux_hidden_states_")]
        for key in aux_keys:
            tensors.pop(key, None)


def _owns_prefill_pp(scheduler):
    return (
        is_hcu()
        and scheduler.disaggregation_mode == DisaggregationMode.PREFILL
        and get_parallel().pp_size > 1
    )


def _dispatch(implementation, original, scheduler, *args, **kwargs):
    if not _owns_prefill_pp(scheduler):
        return original(scheduler, *args, **kwargs)
    return implementation(scheduler, *args, **kwargs)


def around_run_batch(original, scheduler, batch, pp_proxy_tensors=None):
    if not _owns_prefill_pp(scheduler):
        return original(scheduler, batch, pp_proxy_tensors)
    is_dspark = batch.spec_algorithm.is_dspark()
    if is_dspark:
        scheduler.model_worker.set_pp_proxy_tensors_for_next_forward(pp_proxy_tensors)
    try:
        result = original(scheduler, batch, pp_proxy_tensors)
    finally:
        if is_dspark:
            scheduler.model_worker.set_pp_proxy_tensors_for_next_forward(None)
    scheduler._pp_maybe_send_dspark_owner_direct_hidden(batch, result)
    return result


def around_prepare_tensor_dict(original, scheduler, result, batch):
    tensors = original(scheduler, result, batch)
    if (
        _owns_prefill_pp(scheduler)
        and get_pd_hidden_capture_layer_ids(batch.reqs)
        and not scheduler._pp_should_owner_direct_pd_hidden(batch)
        and result.logits_output is not None
        and result.logits_output.hidden_states is not None
    ):
        tensors["pd_aux_hidden_states_0"] = result.logits_output.hidden_states
    return tensors


def around_prep_batch_result(original, scheduler, batch, mb_metadata, pp_outputs):
    result = original(scheduler, batch, mb_metadata, pp_outputs)
    if not _owns_prefill_pp(scheduler):
        return result
    if scheduler._pp_spec_relay and "spec_accept_lens" in pp_outputs.tensors:
        return result
    auxiliary = {
        key: value
        for key, value in pp_outputs.tensors.items()
        if key.startswith("pd_aux_hidden_states_")
    }
    if auxiliary:
        result.pp_hidden_states_proxy_tensors = PPProxyTensors(auxiliary)
    return result


def register_hooks():
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    if hasattr(SchedulerPPMixin, "_pp_pd_get_bootstrapped_ids"):
        return
    for name in (
        "_pp_pd_get_bootstrapped_ids",
        "_pp_should_owner_direct_pd_hidden",
        "_pp_maybe_send_dspark_owner_direct_hidden",
        "_pp_strip_pd_aux_hidden_from_proxy",
    ):
        setattr(SchedulerPPMixin, name, globals()[name])
    SchedulerPPMixin._pp_ordered_union = staticmethod(_pp_ordered_union)
    SchedulerPPMixin._pp_ordered_intersection = staticmethod(_pp_ordered_intersection)
    prefix = "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin."
    for name in (
        "event_loop_pp_disagg_prefill",
        "process_bootstrapped_queue",
        "_pp_pd_get_prefill_transferred_ids",
    ):
        HookRegistry.register(
            prefix + name,
            partial(_dispatch, globals()[name]),
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.run_batch",
        around_run_batch,
        HookType.AROUND,
    )
    HookRegistry.register(
        prefix + "_pp_prepare_tensor_dict",
        around_prepare_tensor_dict,
        HookType.AROUND,
    )
    HookRegistry.register(
        prefix + "_pp_prep_batch_result",
        around_prep_batch_result,
        HookType.AROUND,
    )
