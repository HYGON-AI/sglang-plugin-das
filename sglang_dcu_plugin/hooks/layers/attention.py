"""Hooks for layers/attention."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sglang_dcu_plugin.layers.attention.fla.linear_attention import (
    replacement_chunk_fwd_o,
    replacement_chunk_gated_delta_rule_fwd_h,
    replacement_fused_recurrent_gated_delta_rule_packed_decode,
    replacement_triton_gdn_target_verify,
)
from sglang_dcu_plugin.layers.attention.linear.gdn_backend import (
    replacement_forward_decode,
    replacement_forward_extend,
)


logger = logging.getLogger(__name__)


def register() -> None:
    hooks = (
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_decode",
            replacement_forward_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_extend",
            replacement_forward_extend,
        ),
        (
            "sglang.srt.layers.attention.fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h",
            replacement_chunk_gated_delta_rule_fwd_h,
        ),
        (
            "sglang.srt.layers.attention.fla.chunk_o.chunk_fwd_o",
            replacement_chunk_fwd_o,
        ),
        (
            "sglang.srt.layers.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode",
            replacement_fused_recurrent_gated_delta_rule_packed_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify",
            replacement_triton_gdn_target_verify,
        ),
    )

    for target, hook in hooks:
        logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", target)
        HookRegistry.register(target, hook, HookType.REPLACE)
