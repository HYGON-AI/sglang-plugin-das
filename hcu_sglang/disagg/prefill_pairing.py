from __future__ import annotations

import time


def around_negotiate(original, delayer, prev_state, *args, **kwargs):
    from sglang.srt.environ import envs
    enabled = envs.SGLANG_HCU_PREFILL_MIXED_TIMEOUT.get()
    if enabled and not delayer._queue_trigger_enabled:
        raise ValueError('Mixed prefill timeout requires --prefill-delayer-queue-min-ratio 0')
    result = original(delayer, prev_state, *args, **kwargs)
    if (not enabled or result.input_estimation != 'mixed' or result.output_allow
            or delayer._global_info_buffer[:, 0, 5].max().item() == 0):
        return result
    return result._replace(
        next_state=None, output_allow=True, output_reason='wait_timeout',
        wait_forward_passes=prev_state.delayed_count if prev_state else 0,
        wait_seconds=time.perf_counter()-prev_state.start_time if prev_state else 0.0,
    )
