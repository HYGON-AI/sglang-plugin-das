"""Host-only policy for the experimental Qwen4 eager prefill path.

Keep this module free of torch imports: the scheduler uses the existing mode
all-gather to make the same decision on every rank, including idle ranks.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def prefill_option(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def pure_prefill_modes(modes, *, extend, idle) -> bool:
    """True only for EXTEND + IDLE, never MIXED/verify/draft/decode."""
    return bool(modes) and extend in modes and all(m in (extend, idle) for m in modes)


def compact_prefill_counts(counts, *, pure_prefill, eligible):
    """Preserve active token counts; give idle ranks one reserved dummy row.

    One row avoids zero-size quant/collective APIs and keeps every EP rank in
    the forward. It is a transport row, not an admitted request or cache hit.
    """
    if not eligible or not pure_prefill or len(counts) <= 1:
        return None
    if any(n < 0 for n in counts):
        raise ValueError("negative DP token count")
    if not any(counts):
        return None
    return [max(int(n), 1) for n in counts]
