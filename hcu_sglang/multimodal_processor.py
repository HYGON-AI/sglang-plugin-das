"""Compatibility hooks for the generic Transformers multimodal processor."""

from __future__ import annotations

from typing import Any, Callable

from hcu_sglang.utils.detection import is_hcu


def after_build_mm_items(
    items: list[Any], processor: Any, processor_output: dict, input_ids: Any
) -> list[Any]:
    del processor_output
    from sglang.srt.managers.schedule_batch import Modality

    multi_images = getattr(Modality, "MULTI_IMAGES", None)
    if multi_images is None:
        return items

    image_token_id = processor.mm_tokens.image_token_id
    if image_token_id is None:
        return items
    for item in items:
        if item.modality == multi_images:
            item.offsets = processor.get_mm_items_offset(input_ids, image_token_id)
    return items


async def around_process_mm_data_async(
    original: Callable[..., Any], *args, **kwargs
) -> Any:
    output = await original(*args, **kwargs)
    if output is None:
        return None

    token_type_ids = getattr(output, "token_type_ids", None)
    if token_type_ids is not None and not isinstance(token_type_ids, list):
        tolist = getattr(token_type_ids, "tolist", None)
        if tolist is not None:
            output.token_type_ids = tolist()
    return output


def after_glm5next_rot_pos_emb(result, vision_model, *args, **kwargs):
    from sglang.srt.environ import envs

    if not is_hcu() or envs.SGLANG_ROPE_CACHE_FP32.get():
        return result
    cos, sin, pos_ids = result
    dtype = vision_model.rotary_pos_emb.dtype
    return cos.to(dtype=dtype), sin.to(dtype=dtype), pos_ids


def register_hooks() -> None:
    from hcu_sglang.multimodal import glm5next_processor
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    glm5next_processor.register_hooks()
    HookRegistry.register(
        "sglang.srt.models.glm5_next.Glm5NextVisionModel.rot_pos_emb",
        after_glm5next_rot_pos_emb,
        HookType.AFTER,
    )
    target = (
        "sglang.srt.multimodal.processors.transformers_auto."
        "TransformersAutoMultimodalProcessor"
    )
    HookRegistry.register(
        f"{target}._build_mm_items", after_build_mm_items, HookType.AFTER
    )
    HookRegistry.register(
        f"{target}.process_mm_data_async",
        around_process_mm_data_async,
        HookType.AROUND,
    )
