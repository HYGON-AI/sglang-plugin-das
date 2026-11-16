from __future__ import annotations


def drain_mode():
    from sglang.srt.environ import envs
    mode = envs.SGLANG_HCU_PD_PREFILL_DRAIN_MODE.get()
    if mode not in ('off', 'ready', 'wait'):
        raise ValueError('Prefill drain mode must be off, ready or wait')
    return mode


def drain_previous_result(scheduler, mode, completion_status):
    if mode == 'off' or not scheduler.last_batch or not scheduler.result_queue:
        return False
    if mode == 'ready' and not scheduler.is_disagg_prefill_batch_result_ready(
        scheduler.result_queue[0][1], completion_status
    ):
        return False
    oldest_batch, oldest_result = scheduler.result_queue.popleft()
    scheduler.process_batch_result(oldest_batch, oldest_result)
    scheduler.process_disagg_prefill_inflight_queue()
    scheduler.maybe_supplemental_poll_final_chunks()
    return True
