"""HCU MLA attention backend."""

from __future__ import annotations


class HCUMLABackend:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Migrate from sglang.srt.layers.attention.hcu_mla_backend (Phase 1)."
        )


def register_hooks() -> None:
    pass
