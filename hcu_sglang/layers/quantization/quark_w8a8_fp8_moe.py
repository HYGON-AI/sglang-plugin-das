# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu


def register_hooks() -> None:
    from sglang.srt.layers.quantization.quark.schemes import (
        quark_w8a8_fp8_moe as native,
    )

    native._is_hcu = is_hcu()
    if native._is_hcu:
        native._use_aiter = False
