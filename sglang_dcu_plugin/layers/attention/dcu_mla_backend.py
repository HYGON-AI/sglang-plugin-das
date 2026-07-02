"""DCU MLA attention backend."""

from __future__ import annotations


class DCUMLABackend:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Migrate from sglang.srt.layers.attention.dcu_mla_backend (Phase 1)."
        )


def register_hooks() -> None:
    pass
