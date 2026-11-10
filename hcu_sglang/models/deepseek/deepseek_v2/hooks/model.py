# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Initialize a DSA batch's layer-split cache before its first PP layer."""

from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu


def around_forward(original, self, input_ids, positions, forward_batch, *args, **kwargs):
    if self.use_dsa and self.start_layer < self.end_layer:
        from hcu_sglang.dsa_utils import maybe_prefetch_full_attention_kv

        # The next-layer callback runs after attention. The first layer must
        # replace the previous batch's compact map before writing or reading KV;
        # a PP stage can also start on a layer that reuses the previous TopK.
        maybe_prefetch_full_attention_kv(forward_batch, self.start_layer)
    return original(self, input_ids, positions, forward_batch, *args, **kwargs)


def register_hooks() -> int:
    if not is_hcu():
        return 0
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.deepseek_v2.DeepseekV2Model.forward",
        around_forward,
        HookType.AROUND,
    )
    return 1
