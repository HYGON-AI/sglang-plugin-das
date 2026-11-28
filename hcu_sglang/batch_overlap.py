"""HCU single-batch-overlap enablement."""

from __future__ import annotations

from typing import Callable

from hcu_sglang.utils.detection import is_hcu


def around_enable_combine_down_gemm(original: Callable[..., bool], cls: type) -> bool:
    if is_hcu():
        from sglang.srt.layers.moe.utils import is_sbo_enabled

        return is_sbo_enabled()
    return original(cls)


def around_compute_overlap_args(original, dispatch_output, alt_stream):
    if is_hcu():
        import torch

        hidden_states = dispatch_output.hidden_states
        if hidden_states.dtype in (
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ) or hidden_states.ndim != 3:
            return None, None, {}
    return original(dispatch_output, alt_stream)


def around_dsv41_prefill_admission(original, self, running_batch):
    """Free completed request slots before CP prefill admission in the overlap loop."""
    if running_batch.reqs and self.enable_skip_finishing_decode:
        from sglang.srt.environ import envs

        args = self.server_args
        if (
            args.enable_prefill_cp
            and args.dp_size == args.pp_size == 1
            and self.spec_algorithm.is_none()
            and self.model_config.hf_config.model_type == "deepseek_v41"
            and is_hcu()
            and envs.SGLANG_DSV41_CP_PRUNE_FINISHING_PREFILL.get()
        ):
            pending = []
            if self.result_queue and self.result_queue[0][0].forward_mode.is_extend():
                pending = [
                    req for req in self._reqs_finishing_inflight(running_batch)
                    if req.inflight_middle_chunks <= 0
                ]
            if pending or any(req.finished() for req in running_batch.reqs):
                # The result queue owns a separate batch snapshot and releases KV
                # when its sample is committed. Only prune admission bookkeeping.
                running_batch.filter_batch(reqs_to_exclude=pending)
                running_batch.batch_is_full = False
    return original(self, running_batch)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap."
        "SboFlags.enable_combine_down_gemm_two_stream_overlap",
        around_enable_combine_down_gemm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap.compute_overlap_args",
        around_compute_overlap_args,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill",
        around_dsv41_prefill_admission,
        HookType.AROUND,
    )
