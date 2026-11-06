# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from hcu_sglang.glm5_next_policy import is_glm5_next_hcu
from hcu_sglang.utils.detection import is_hcu


def around_create_dsa_backend(original, runner):
    if is_hcu() and is_glm5_next_hcu(runner.model_config.hf_config):
        from hcu_sglang.layers.attention.glm5_next.dsa_backend import (
            NativeSparseAttnBackend,
        )

        return NativeSparseAttnBackend(runner)
    return original(runner)


def register() -> None:
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.attention_registry.create_dsa_backend",
        around_create_dsa_backend,
        HookType.AROUND,
    )
