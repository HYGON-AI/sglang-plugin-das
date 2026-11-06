# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_FAMILIES = {
    "muse_spark": (
        "hcu_sglang.models.muse_spark.entries",
        "hcu_sglang.models.muse_spark.",
        frozenset({"MuseSparkForConditionalGeneration"}),
    ),
    "glm5_next": (
        "hcu_sglang.models.glm5_next.entries",
        "hcu_sglang.models.glm5_next.",
        frozenset({
            "Glm5NextForCausalLM",
            "Glm5NextForConditionalGeneration",
            "Glm5NextForCausalLMNextN",
            "Glm5NextForConditionalGenerationNextN",
        }),
    ),
}


def around_resolve_model_cls(original, registry, architectures):
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return original(registry, architectures)
    requested = [architectures] if isinstance(architectures, str) else architectures
    for package, prefix, supported in _FAMILIES.values():
        selected = supported.intersection(requested)
        if not selected:
            continue
        if not all(
            getattr(registry.models.get(name), "__module__", "").startswith(prefix)
            for name in selected
        ):
            registry.register(package, overwrite=True, strict=True)
        for name in selected:
            if not getattr(registry.models.get(name), "__module__", "").startswith(prefix):
                raise RuntimeError(f"Owned model architecture {name} was not registered")
    return original(registry, architectures)


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.registry._ModelRegistry.resolve_model_cls",
        around_resolve_model_cls,
        HookType.AROUND,
    )
