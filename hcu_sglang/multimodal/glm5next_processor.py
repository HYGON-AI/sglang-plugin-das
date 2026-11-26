# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import logging
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

_GLM5NEXT_MODEL_TYPES = frozenset({"glm5_next", "glm5v_next"})
# From the video_preprocessor_config.json shipped with the GLM-5-Next processor code.
_GLM5NEXT_VIDEO_MAX_FRAME_COUNT = 2048


def _read_processor_config(path, revision):
    from sglang.srt.utils.hf_transformers.common import _resolve_local_or_cached_file

    try:
        filename = _resolve_local_or_cached_file(
            path, "processor_config.json", revision
        )
        with open(filename) as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def _find_glm5next_legacy_processor_config(*, tokenizer_name, model_name, revision):
    """Return ``(path, config)`` of a nested processor_config.json without auto_map.

    Transformers reads the nested sections before preprocessor_config.json and
    cannot resolve ``Glm5NextImageProcessor`` without remote code, so these
    checkpoints load as tokenizer-only. The tokenizer path is checked first.
    """
    from sglang.srt.utils.hf_transformers.common import _cached_file_exists

    candidates = []
    for candidate in (tokenizer_name, model_name):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        processor_config = _read_processor_config(candidate, revision)
        image_config = processor_config.get("image_processor")
        if isinstance(image_config, dict):
            if "auto_map" in processor_config or "auto_map" in image_config:
                # The checkpoint ships its own processor code; Transformers loads it.
                return None, None
            return candidate, processor_config
        if _cached_file_exists(candidate, "preprocessor_config.json", revision):
            # This path defines its processor the standard way; Transformers loads it.
            return None, None
    return None, None


def _build_glm5next_image_processor(*, image_config):
    from .glm5next.image_processing_glm5next import Glm5nextImageProcessorFast

    image_kwargs = {
        key: value
        for key, value in image_config.items()
        if key != "image_processor_type"
    }
    # Fast class for every backend, as with the checkpoint code, whose auto_map
    # names only Glm5nextImageProcessorFast.
    image_processor = Glm5nextImageProcessorFast(**image_kwargs)
    adapt_glm5next_image_processor(image_processor)
    return image_processor


def _build_glm5next_video_processor(*, video_config):
    from .glm5next.video_processing_glm5next import Glm5nextVideoProcessor

    video_kwargs = {
        key: value
        for key, value in video_config.items()
        if key != "video_processor_type"
    }
    # The legacy config records only fps, while sample_frames reads both of these.
    video_kwargs.setdefault(
        "fps_interval", video_kwargs.get("fps", Glm5nextVideoProcessor.fps)
    )
    video_kwargs.setdefault("max_frame_count_dynamic", _GLM5NEXT_VIDEO_MAX_FRAME_COUNT)
    return Glm5nextVideoProcessor(**video_kwargs)


def adapt_glm5next_image_processor(processor):
    """Keep checkpoint fast preprocessing compatible with HF's resample API."""
    image_processor = getattr(processor, "image_processor", processor)
    if type(image_processor).__name__ != "Glm5nextImageProcessorFast":
        return
    cls = type(image_processor)
    original = cls._preprocess
    if not getattr(original, "_sglang_resample_compat", False) and (
        "interpolation" in inspect.signature(original).parameters
    ):

        @wraps(original)
        def preprocess(self, *args, **kwargs):
            if "interpolation" not in kwargs:
                from transformers.image_utils import pil_torch_interpolation_mapping

                resample = kwargs.get("resample", self.resample)
                kwargs["interpolation"] = pil_torch_interpolation_mapping.get(
                    resample, resample
                )
            return original(self, *args, **kwargs)

        preprocess._sglang_resample_compat = True
        cls._preprocess = preprocess

    # Glm5nextImageProcessorFast was written against the old HF fast backend,
    # whose resize keyword is ``interpolation``. Transformers 5.12's
    # TorchvisionBackend renamed that keyword to ``resample``. Without this
    # shim the keyword is swallowed by **kwargs and resize silently falls back
    # to bilinear interpolation, changing the image tensor while remaining on
    # the fast path.
    resize = getattr(cls, "resize", None)
    if resize is not None:
        resize_signature = inspect.signature(resize)
        if (
            "resample" in resize_signature.parameters
            and "interpolation" not in resize_signature.parameters
            and not getattr(resize, "_sglang_resize_compat", False)
        ):
            resize_method = resize

            @wraps(resize_method)
            def resize_compat(self, *args, **kwargs):
                if "interpolation" in kwargs and "resample" not in kwargs:
                    kwargs["resample"] = kwargs.pop("interpolation")
                return resize_method(self, *args, **kwargs)

            resize_compat._sglang_resize_compat = True
            cls.resize = resize_compat


def _build_media_processors(processor_path, processor_config):
    image_processor = _build_glm5next_image_processor(
        image_config=processor_config["image_processor"]
    )
    video_config = processor_config.get("video_processor")
    video_processor = (
        _build_glm5next_video_processor(video_config=video_config)
        if isinstance(video_config, dict) and video_config
        else None
    )
    logger.info("Using bundled GLM-5-Next legacy processors from %s", processor_path)
    return image_processor, video_processor


def _load_encoder_processors(original, preprocessor, server_args):
    from sglang.srt.runtime_context import get_model, get_serving

    if preprocessor.model_type not in _GLM5NEXT_MODEL_TYPES:
        return original(preprocessor, server_args)

    model = get_model()
    tokenizer_path = get_serving().tokenizer_path or model.model_path
    path, config = _find_glm5next_legacy_processor_config(
        tokenizer_name=tokenizer_path, model_name=model.model_path, revision=model.revision
    )
    if config is None:
        result = original(preprocessor, server_args)
        adapt_glm5next_image_processor(preprocessor.image_processor)
        return result

    preprocessor.image_processor, preprocessor.video_processor = _build_media_processors(
        path, config
    )
    preprocessor.audio_processor = None

    from transformers import AutoProcessor

    try:
        audio_processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model.trust_remote_code,
            revision=model.revision,
        )
    except Exception as exc:
        logger.warning("Failed to load audio processor: %s", exc)
    else:
        if hasattr(audio_processor, "feature_extractor"):
            preprocessor.audio_processor = audio_processor


def _get_processor(
    original,
    tokenizer_name,
    *args,
    tokenizer_mode="auto",
    trust_remote_code=False,
    tokenizer_revision=None,
    use_fast=None,
    image_processor_backend=None,
    tokenizer_backend="huggingface",
    model_name=None,
    **kwargs,
):
    from sglang.srt.utils.hf_transformers.common import (
        AutoConfig,
        download_from_hf,
        resolve_runai_obj_uri,
    )

    def fallback():
        processor = original(
            tokenizer_name,
            *args,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            use_fast=use_fast,
            image_processor_backend=image_processor_backend,
            tokenizer_backend=tokenizer_backend,
            model_name=model_name,
            **kwargs,
        )
        adapt_glm5next_image_processor(processor)
        return processor

    revision = kwargs.get("revision", tokenizer_revision)
    tokenizer_path = resolve_runai_obj_uri(tokenizer_name)
    model_path = resolve_runai_obj_uri(model_name) if model_name is not None else None
    path, config = _find_glm5next_legacy_processor_config(
        tokenizer_name=tokenizer_path, model_name=model_path, revision=revision
    )
    if config is None or config.get("processor_class") not in {
        "Glm5NextProcessor",
        "Glm5nextProcessor",
    }:
        return fallback()

    hf_config = AutoConfig.from_pretrained(
        model_path or tokenizer_path, trust_remote_code=trust_remote_code, revision=revision
    )
    if hf_config.model_type not in _GLM5NEXT_MODEL_TYPES or getattr(
        hf_config, "language_model_only", False
    ):
        return fallback()

    from sglang.srt.utils.hf_transformers.processor import _normalize_image_processor_backend
    from sglang.srt.utils.hf_transformers.tokenizer import (
        _fix_added_tokens_encoding,
        get_tokenizer,
    )

    from .glm5next.processing_glm5next import Glm5nextProcessor

    _normalize_image_processor_backend(image_processor_backend, use_fast)
    image_processor, video_processor = _build_media_processors(path, config)
    tokenizer_kwargs = dict(kwargs)
    tokenizer_kwargs.pop("revision", None)
    tokenizer = get_tokenizer(
        tokenizer_path,
        *args,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=revision,
        tokenizer_backend=tokenizer_backend,
        **tokenizer_kwargs,
    )
    if tokenizer.chat_template is None:
        local_path = download_from_hf(
            tokenizer_path, allow_patterns=["*.json", "*.jinja", "*.model"]
        )
        jinja_path = Path(local_path) / "chat_template.jinja"
        if jinja_path.is_file():
            tokenizer.chat_template = jinja_path.read_text()
    _fix_added_tokens_encoding(tokenizer)
    return Glm5nextProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.utils.hf_transformers import processor

    if hasattr(processor, "load_glm5next_legacy_media_processors"):
        return
    HookRegistry.register(
        "sglang.srt.disaggregation.encoder.preprocessor.EncoderPreprocessor._load_mm_processor",
        _load_encoder_processors,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.utils.hf_transformers.processor.get_processor",
        _get_processor,
        HookType.AROUND,
    )
