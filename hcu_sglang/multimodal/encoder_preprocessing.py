# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.multimodal.encoder_preprocessing import (
    EncoderMediaProcessorConfig as NativeEncoderMediaProcessorConfig,
)


@dataclass(frozen=True)
class EncoderMediaProcessorConfig(NativeEncoderMediaProcessorConfig):
    preprocess_off_event_loop: bool = False
    audio_sample_rate: int | None = None


def register_hooks() -> None:
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.multimodal.encoder_preprocessing.EncoderMediaProcessorConfig",
        EncoderMediaProcessorConfig,
        HookType.REPLACE,
    )
