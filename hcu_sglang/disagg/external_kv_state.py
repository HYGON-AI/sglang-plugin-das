# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_sglang.utils.detection import is_hcu


def after_init_chunked_prefill(result, scheduler):
    if not is_hcu():
        return result
    for name, factory in (
        ("_external_kv_abort_rids", set),
        ("_external_kv_cleanup_rids", set),
        ("_external_kv_chunked_abort_reqs", dict),
    ):
        if not hasattr(scheduler, name):
            setattr(scheduler, name, factory())
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if not is_hcu():
        return
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.init_chunked_prefill",
        after_init_chunked_prefill,
        HookType.AFTER,
    )
