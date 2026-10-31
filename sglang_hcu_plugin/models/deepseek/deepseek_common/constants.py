"""HCU-specific module-level flags for deepseek_common.utils patch."""

from __future__ import annotations

from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()

MODULE_PATCH_ATTRS = ("_is_hcu",)
