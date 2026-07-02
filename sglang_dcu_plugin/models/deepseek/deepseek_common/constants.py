"""DCU-specific module-level flags for deepseek_common.utils patch."""

from __future__ import annotations

from sglang.srt.utils import is_dcu

_is_dcu = is_dcu()

MODULE_PATCH_ATTRS = ("_is_dcu",)
