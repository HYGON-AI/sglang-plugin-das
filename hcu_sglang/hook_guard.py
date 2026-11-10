# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep HCU hook registration from flooding startup logs with tracebacks.

SGLang resolves every registered target lazily inside
``HookRegistry._apply_target()`` and ``apply_hooks()`` logs a full traceback per
unresolvable target. The HCU plugin registers hooks for optional surfaces --
multimodal / video-generation pipelines, CuTeDSL-only kernels and upstream
private helpers that get renamed between revisions -- so one startup can emit
hundreds of tracebacks whose only meaning is "this optional target is not
present in the current build".

Measured on the 2026-11-09 Qwen3.8-Flash-Next PD run against official v0.5.22:
226 tracebacks, 210 of them ``Failed to apply hooks to <target>`` over 15
targets x 7 processes. 14 of those 15 targets are multimodal/video-generation
surfaces that a text model never loads.

This module wraps ``HookRegistry._apply_target`` once per process so that an
unresolvable target is skipped and reported once in a summary line at INFO,
instead of one traceback per target per process.

``HCU_HOOK_GUARD=0`` restores the unguarded behaviour for debugging.

Why wrap ``_apply_target`` instead of ``register``: resolution is attempted
first and only a *failed* attempt is diverted. Every target that does resolve
takes the original code path unchanged, so the guard cannot skip a hook that
would otherwise have been applied -- which a pre-resolution filter could, since
some target modules only become importable after every plugin has registered.
Registration itself is untouched, so ``HookRegistry._hooks`` still lists exactly
what the plugin declared and the target-contract tooling keeps working.

Scope note: this guard deliberately does not touch SGLang's "Multiple AROUND
hooks" notice. Several HCU targets carry two AROUND wrappers that self-gate at
call time on mutually exclusive runner backends (``w8a8_lightop`` vs
``w8a8_aiter``); dropping either one at registration time would permanently
disable that quantization path rather than merely skip a branch. Removing that
notice requires merging each pair into a single dispatching wrapper, which
changes quantization code and needs device validation -- tracked separately.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Target -> exception type name that made it unresolvable, in encounter order.
_skipped: dict[str, str] = {}
_installed = False


def _report() -> None:
    """Emit one summary line per failure kind for targets this build lacks."""
    if not _skipped:
        return
    grouped: dict[str, list[str]] = {}
    for target, reason in _skipped.items():
        grouped.setdefault(reason, []).append(target)
    for reason, targets in grouped.items():
        logger.info(
            "hcu hook guard: skipped %d hook target(s) absent in this SGLang "
            "build (%s): %s",
            len(targets),
            reason,
            ", ".join(targets),
        )
    _skipped.clear()


def install() -> None:
    """Install the guard once per process; no-op when disabled."""
    global _installed
    if _installed:
        return
    if os.environ.get("HCU_HOOK_GUARD", "1") == "0":
        logger.info("hcu hook guard disabled by HCU_HOOK_GUARD=0")
        return

    from sglang.srt.plugins import hook_registry

    registry = hook_registry.HookRegistry
    # Both members are classmethods in SGLang, so attribute access yields a
    # *bound* method. Reach through __func__ for the raw callable, otherwise
    # re-invoking with cls would bind it twice.
    raw_apply_target = registry.__dict__["_apply_target"].__func__
    raw_apply_hooks = registry.__dict__["apply_hooks"].__func__
    if getattr(raw_apply_target, "_hcu_guarded", False):
        return

    def guarded_apply_target(cls, target, hooks):
        try:
            return raw_apply_target(cls, target, hooks)
        except Exception as exc:
            _skipped[target] = type(exc).__name__
            logger.debug("hcu hook guard: skip %s (%s)", target, exc)
            return None

    def guarded_apply_hooks(cls):
        try:
            return raw_apply_hooks(cls)
        finally:
            _report()

    guarded_apply_target._hcu_guarded = True
    registry._apply_target = classmethod(guarded_apply_target)
    registry.apply_hooks = classmethod(guarded_apply_hooks)
    _installed = True
    logger.info("hcu hook guard installed")
