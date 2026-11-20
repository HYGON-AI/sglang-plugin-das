import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .minimax_m31_dspark import _minimax_hip_dspark


@dataclass
class _HostMirror:
    buffer: torch.Tensor
    event: object
    ready: bool = False

    def read(self, indices):
        self.event.synchronize()
        lengths = self.buffer[indices]
        return lengths, int(lengths.sum())


@dataclass
class _DecodeStep:
    scheduler: object
    batch: object
    mirror: _HostMirror | None = None
    lengths: tuple[torch.Tensor, int] | None = None


_decode_step = ContextVar("minimax_m31_decode_seq_lens", default=None)


def _enabled(scheduler):
    if os.getenv("SGLANG_HCU_ASYNC_SEQ_LENS_D2H", "1") != "1" or not _minimax_hip_dspark():
        return False

    from sglang.srt.configs.model_config import is_minimax_sparse
    from sglang.srt.managers.overlap_utils import _DEBUG_ASSERT
    from sglang.srt.runtime_context import get_parallel

    return (
        not _DEBUG_ASSERT
        and is_minimax_sparse(scheduler.model_config.hf_config)
        and get_parallel().pp_size == 1
    )


def _publish(original, relay, future_indices, new_seq_lens, confidence=None):
    result = original(relay, future_indices, new_seq_lens, confidence)
    mirror = getattr(relay, "_m31_seq_lens_mirror", None)
    if mirror is not None:
        mirror.ready = False
    if (
        not relay.needs_cpu_seq_lens
        or relay.needs_confidence_relay
        or future_indices.numel() == 0
        or not relay.new_seq_lens_buf.is_cuda
        or not getattr(relay, "_m31_async_seq_lens", False)
        or torch.cuda.is_current_stream_capturing()
    ):
        return result

    if mirror is None:
        mirror = relay._m31_seq_lens_mirror = _HostMirror(
            torch.empty(
                relay.new_seq_lens_buf.shape,
                dtype=relay.new_seq_lens_buf.dtype,
                device="cpu",
                pin_memory=True,
            ),
            torch.get_device_module(relay.device).Event(),
        )
    mirror.buffer.copy_(relay.new_seq_lens_buf, non_blocking=True)
    mirror.event.record()
    mirror.ready = True
    return result


def defer_resolution(relay, batch, stream_ordered):
    mirror = getattr(relay, "_m31_seq_lens_mirror", None)
    if mirror is None or not mirror.ready:
        return False
    mirror.ready = False
    step = _decode_step.get()
    indices = batch.req_pool_indices_cpu
    future_indices = getattr(batch.spec_info, "future_indices", None)
    if (
        stream_ordered
        or step is None
        or step.batch is not batch
        or step.scheduler.future_map is not relay
        or not isinstance(indices, torch.Tensor)
        or indices.device.type != "cpu"
        or indices.ndim != 1
        or indices.dtype != torch.int64
        or future_indices is None
        or not indices.numel()
        or indices.numel() != future_indices.numel()
    ):
        return False
    step.mirror = mirror
    relay._publish_fresh = False
    return True


def _run_batch(original, scheduler, batch, *args, **kwargs):
    if not scheduler.enable_overlap:
        return original(scheduler, batch, *args, **kwargs)
    enabled = _enabled(scheduler)
    scheduler.future_map._m31_async_seq_lens = enabled
    if (
        not enabled
        or not batch.spec_algorithm.is_dspark()
        or not batch.forward_mode.is_decode()
        or batch.is_extend_in_batch
        or batch.prefill_input_ids_cpu is not None
        or scheduler._confidence_budget_prepare is not None
    ):
        return original(scheduler, batch, *args, **kwargs)

    step = _DecodeStep(scheduler, batch)
    token = _decode_step.set(step)
    try:
        result = original(scheduler, batch, *args, **kwargs)
        if step.mirror is not None:
            if step.lengths is None:
                raise RuntimeError("MiniMax DSpark did not consume its published host lengths")
            batch.seq_lens_cpu, batch.seq_lens_sum = step.lengths
        return result
    finally:
        _decode_step.reset(token)


@contextmanager
def _forward_isolation(original, scheduler, batch, *, overlap):
    step = _decode_step.get()
    if step is not None and step.batch is batch and step.mirror is not None:
        relay = scheduler.future_map
        batch.seq_lens = relay.new_seq_lens_buf[batch.spec_info.future_indices]
    with original(scheduler, batch, overlap=overlap):
        yield


def consume_published_lengths(batch):
    step = _decode_step.get()
    if step is not None and step.batch is batch and step.mirror is not None:
        if step.lengths is None:
            step.lengths = step.mirror.read(batch.req_pool_indices_cpu)
        batch.seq_lens_cpu, batch.seq_lens_sum = step.lengths


def _forward_decode(original, worker, batch, *args, **kwargs):
    consume_published_lengths(batch)
    return original(worker, batch, *args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        ("managers.overlap_utils.FutureMap.publish", _publish),
        ("managers.scheduler.Scheduler.run_batch", _run_batch),
        ("managers.scheduler.Scheduler._forward_isolation", _forward_isolation),
        ("speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2._forward_decode", _forward_decode),
    ):
        HookRegistry.register("sglang.srt." + target, hook, HookType.AROUND)
