from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu


def _layout_enabled(processor, mode):
    from sglang.srt.runtime_context import get_disagg, get_mm

    if (
        processor is None
        or type(processor).__name__ != "Glm4vImageProcessor"
        or processor.hf_config.model_type != "glm5_next"
        or getattr(processor, "builds_epd_layout_from_media", False)
    ):
        return False
    disagg = get_disagg()
    return (
        disagg.disaggregation_mode == mode
        and disagg.language_only
        and not disagg.language_model_only
        and disagg.encoder_transfer_backend == "mooncake"
        and disagg.disaggregation_transfer_backend == "mooncake"
        and not get_mm().mm_process_config
    )


def _decode_prompt_ids(processor, request):
    from sglang.srt.environ import envs
    from sglang.srt.managers.io_struct import GenerateReqInput

    if not _layout_enabled(processor, "decode") or not isinstance(
        request, GenerateReqInput
    ):
        return None
    images = request.image_data
    if (
        not request.is_single
        or request.batch_size != 1
        or request.bootstrap_room is None
        or request.input_ids is not None
        or request.input_embeds is not None
        or request.positional_embed_overrides is not None
        or request.mm_hashes is not None
        or request.mm_content_hashes is not None
        or request.session_id is not None
        or request.session_params is not None
        or request.video_data
        or request.audio_data
        or not isinstance(request.text, str)
        or not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0].startswith("data:image/jpeg;base64,")
        or processor.prefer_tokenized_input
        or type(getattr(processor._processor, "image_processor", None)).__name__
        != "Glm5nextImageProcessorFast"
        or envs.SGLANG_MM_SKIP_COMPUTE_HASH.get()
    ):
        return None

    prompt_ids = processor._tokenizer.encode(request.text)
    image_marker = processor.IM_TOKEN_ID
    if prompt_ids.count(image_marker) != 1 or any(
        token in prompt_ids
        for token in (
            processor.VIDEO_TOKEN_ID,
            processor.VIDEO_START_TOKEN_ID,
            processor.VIDEO_END_TOKEN_ID,
        )
    ):
        return None
    index = prompt_ids.index(image_marker)
    if (
        index == 0
        or index + 1 == len(prompt_ids)
        or prompt_ids[index - 1] != processor.IM_START_TOKEN_ID
        or prompt_ids[index + 1] != processor.IM_END_TOKEN_ID
    ):
        return None
    return prompt_ids


def _around_dispatch(original, manager, obj):
    if _decode_prompt_ids(manager.mm_processor, obj) is None:
        return original(manager, obj)
    obj.need_wait_for_mm_inputs = False
    return None


async def _around_glm_image_processor(
    original, processor, image_data, input_text, request_obj, *args, **kwargs
):
    prompt_ids = (
        _decode_prompt_ids(processor, request_obj)
        if getattr(request_obj, "need_wait_for_mm_inputs", True) is False
        else None
    )
    if prompt_ids is None:
        return await original(
            processor, image_data, input_text, request_obj, *args, **kwargs
        )

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalProcessorOutput,
    )

    loaded = await processor.fast_load_mm_data(
        prompt="",
        multimodal_tokens=processor.mm_tokens,
        image_data=image_data,
        return_text=False,
    )
    if len(loaded.images) != 1 or loaded.images[0] is None or isinstance(
        loaded.images[0], dict
    ):
        raise ValueError("Expected one decoded image for NMZ Decode")
    executor = processor.mm_processor_executor
    if executor is None:
        media = processor.process_mm_data(input_text="", images=loaded.images)
    else:
        media = await executor.run(
            processor.process_mm_data, input_text="", images=loaded.images
        )
    grid = media["image_grid_thw"]
    pixels = media["pixel_values"]
    if grid.shape != (1, 3) or pixels.shape[0] != int(grid.prod().item()):
        raise ValueError("Unexpected image grid or patch count for NMZ Decode")
    input_ids, offsets, modalities = processor.build_input_ids(
        prompt_ids, img_grid_thw=grid
    )
    if modalities != [Modality.IMAGE] or len(offsets) != 1:
        raise ValueError("Expected one image span in the NMZ Decode prompt")

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=pixels,
        model_specific_data={"image_grid_thw": grid},
    )
    positions, position_delta = processor.compute_mrope_positions(input_ids, [item])
    item.set_pad_value()
    item.feature = None
    return MultimodalProcessorOutput(
        input_ids=input_ids,
        mm_items=[item],
        im_start_id=processor.IM_START_TOKEN_ID,
        im_end_id=processor.IM_END_TOKEN_ID,
        im_token_id=processor.IM_TOKEN_ID,
        video_token_id=processor.VIDEO_TOKEN_ID,
        mrope_positions=positions,
        mrope_position_delta=position_delta,
    )


def _after_epd_mm_data(output, processor, *args, **kwargs):
    if not _layout_enabled(processor, "prefill"):
        return output

    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    grid = kwargs.get("img_grid_thw")
    if (
        output is None
        or output.mrope_positions is not None
        or grid is None
        or len(grid) != 1
        or len(output.mm_items) != 1
        or output.mm_items[0].modality != Modality.IMAGE
        or kwargs.get("video_grid_thw") is not None
        or kwargs.get("audio_feature_lens") is not None
    ):
        return output
    layout_item = MultimodalDataItem(
        modality=Modality.IMAGE, model_specific_data={"image_grid_thw": grid}
    )
    output.mrope_positions, output.mrope_position_delta = (
        processor.compute_mrope_positions(output.input_ids, [layout_item])
    )
    return output


def register_hooks():
    from sglang.srt.environ import envs

    if not envs.SGLANG_HCU_NMZ_EPD_IMAGE_LAYOUT.get() or not is_hcu():
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.managers.tokenizer_manager.TokenizerManager."
        "_handle_epd_disaggregation_encode_request",
        _around_dispatch,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.multimodal.processors.glm4v.Glm4vImageProcessor.process_mm_data_async",
        _around_glm_image_processor,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor.get_mm_data",
        _after_epd_mm_data,
        HookType.AFTER,
    )
