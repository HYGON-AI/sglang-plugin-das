from __future__ import annotations

import torch

from hcu_sglang.utils.detection import is_hcu


def around_forward(
    original,
    self,
    logits_output,
    sampling_info,
    return_logprob,
    top_logprobs_nums,
    token_ids_logprobs,
    positions,
    *,
    logprob_logits=None,
):
    logits = logits_output.next_token_logits
    if logits.shape[0] == 0 and is_hcu():
        from sglang.srt.layers import sampler as native

        native.maybe_detect_nan(logits, "sampler: next_token_logits")
        native._trace_e2e_sampler(
            "forward_enter",
            logits_shape=tuple(logits.shape),
            all_greedy=sampling_info.is_all_greedy,
        )
        return torch.empty((0,), dtype=torch.int64, device=logits.device)
    return original(
        self,
        logits_output,
        sampling_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
        positions,
        logprob_logits=logprob_logits,
    )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.sampler.Sampler.forward", around_forward, HookType.AROUND
    )
