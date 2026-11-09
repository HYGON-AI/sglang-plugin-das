"""HCU quantization adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.quantization import (
        auto_round,
        fp8,
        gptq,
        kv_cache,
        moe_wna16,
        quark_w8a8_fp8_moe,
        registry,
        unquantized_aiter,
        unquantized_moe_dispatch,
        unquantized_marlin,
        w8a8_aiter,
        w8a8_fp8,
        w8a8_lightop,
        w8a8_runner_dispatch,
    )

    auto_round.register_hooks()
    kv_cache.register_hooks()
    moe_wna16.register_hooks()
    fp8.register_hooks()
    gptq.register_hooks()
    quark_w8a8_fp8_moe.register_hooks()
    w8a8_fp8.register_hooks()
    registry.register_hooks()
    unquantized_moe_dispatch.register_hooks()
    unquantized_marlin.register_hooks()
    w8a8_aiter.register_hooks()
    w8a8_lightop.register_hooks()
    w8a8_runner_dispatch.register_hooks()
    logger.debug("[hcu_sglang] quantization hooks registered")

    from . import compressed_targets
    compressed_targets.register_hooks()

    from . import compressed_w4a8
    compressed_w4a8.register_hooks()
