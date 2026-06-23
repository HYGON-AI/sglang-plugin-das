"""SGLang demo general plugin.

Demonstrates AFTER / AROUND / REPLACE hooks via the unified plugin framework.
Install with: pip install -e examples/demo_general_plugin
Enable with:  export SGLANG_PLUGINS=demo_general
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from sglang.srt.plugins.hook_registry import HookRegistry, HookType, plugin_hook

logger = logging.getLogger(__name__)

# Per-scheduler batch counter for AFTER hook demo (avoid log spam).
_batch_counters: defaultdict[int, int] = defaultdict(int)
_LOG_EVERY_N_BATCHES = 10000


def register() -> None:
    """Entry point called by load_plugins(). Register hooks here."""
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        _server_args_post_init_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.__init__",
        _scheduler_init_hook,
        HookType.AROUND,
    )
    # Capture original in closure (not module global). HookRegistry._propagate_patch
    # replaces any module attribute that still references the patched function object;
    # a module-level binding would become the wrapper and cause infinite recursion.
    original_check_server_args = _resolve_original(
        "sglang.srt.server_args.ServerArgs.check_server_args"
    )

    def _check_server_args_replace(self) -> None:
        logger.info(
            "[demo_general] REPLACE: check_server_args for model_path=%s",
            getattr(self, "model_path", None),
        )
        original_check_server_args(self)

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.check_server_args",
        _check_server_args_replace,
        HookType.REPLACE,
    )
    logger.info("[demo_general] Registered ServerArgs and Scheduler hooks")


def _resolve_original(target: str):
    """Resolve the current callable for *target* (pre-apply_hooks)."""
    from importlib import import_module

    module_path, attr_name = target.rsplit(".", 1)
    parent_path, _, name = module_path.rpartition(".")
    module = import_module(parent_path)
    parent = getattr(module, name)
    fn = getattr(parent, attr_name)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@plugin_hook(
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    type=HookType.AFTER,
)
def _batch_counter_hook(result, self, *args, **kwargs):
    """AFTER demo: log every N scheduling calls."""
    scheduler_id = id(self)
    _batch_counters[scheduler_id] += 1
    count = _batch_counters[scheduler_id]
    if count % _LOG_EVERY_N_BATCHES == 0:
        has_batch = result is not None
        logger.info(
            "[demo_general] get_next_batch_to_run #%d, has_batch=%s",
            count,
            has_batch,
        )
    return None  # keep original return value


def _server_args_post_init_hook(original_fn, self, *args, **kwargs):
    """AROUND demo: log key ServerArgs fields after __post_init__."""
    result = original_fn(self, *args, **kwargs)
    logger.info(
        "[demo_general] ServerArgs initialized: model_path=%s, tp_size=%s, "
        "attention_backend=%s",
        getattr(self, "model_path", None),
        getattr(self, "tp_size", None),
        getattr(self, "attention_backend", None),
    )
    return result


def _scheduler_init_hook(original_fn, self, *args, **kwargs):
    """AROUND demo: measure Scheduler.__init__ latency."""
    start = time.perf_counter()
    result = original_fn(self, *args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[demo_general] Scheduler initialized in %.1f ms (gpu_id=%s)",
        elapsed_ms,
        getattr(self, "gpu_id", None),
    )
    return result
