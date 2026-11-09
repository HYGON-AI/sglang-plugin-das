# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single handle_eplb_and_dispatch hook (load-aware + locality validation)."""

from __future__ import annotations

from typing import Any, Callable


def around_handle_eplb_and_dispatch(
    original: Callable[..., Any], server_args: Any
) -> Any:
    # Both validators are post-original and independent; preserve the stacked
    # order the two independent registrations produced (placement outer,
    # load-aware integration inner).
    from hcu_sglang.eplb import integration, placement

    def chained_original(*args):
        return integration.around_dispatch_validation(original, *args)

    return placement.around_validation(chained_original, server_args)


def register_hooks() -> None:
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch",
        around_handle_eplb_and_dispatch,
        HookType.AROUND,
    )
