"""Hooks for models/qwen."""

import logging
logger = logging.getLogger(__name__)

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_dcu_plugin.models.qwen.qwen3_5 import replacement_self_attention

def register() -> None:
    logger.debug("[sglang_dcu_plugin] qwen hooks: placeholder")

    # qwen3.5 self_attention
    logger.info("Registered Qwen3.5 self_attention REPLACE hook: %s", "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention")
    HookRegistry.register(
        "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention",
        replacement_self_attention,
        HookType.REPLACE,
    )
