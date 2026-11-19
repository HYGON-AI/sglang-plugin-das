import os

from hcu_sglang.layers.quantization.registry import register_hooks

ENTRY_POINT = "hcu_minimax_m31"


def register() -> None:
    selected = {name.strip() for name in os.getenv("SGLANG_PLUGINS", "").split(",")}
    if ENTRY_POINT not in selected or "hcu" in selected:
        return
    if os.getenv("HCU_PLUGIN_MINIMAX_M31_QUANT", "1") == "1":
        from sglang.srt.arg_groups.choices import (
            QUANTIZATION_CHOICES,
            add_quantization_method_choices,
        )

        from hcu_sglang.layers.quantization import compressed_int8
        from hcu_sglang.layers.quantization.registry import SLIMQUANT_MARLIN_NAME

        if SLIMQUANT_MARLIN_NAME not in QUANTIZATION_CHOICES:
            add_quantization_method_choices([SLIMQUANT_MARLIN_NAME])
        register_hooks()
        compressed_int8.register_hooks()

    from hcu_sglang.models import minimax_m31_dspark, minimax_m31_seq_lens

    minimax_m31_dspark.register_hooks()
    minimax_m31_seq_lens.register_hooks()

    if os.getenv("HCU_PLUGIN_MINIMAX_M31_FLASHMLA", "0") == "1":
        from hcu_sglang.layers.attention import (
            minimax_m31_decode_graph,
            minimax_m31_draft_swa,
            minimax_m31_fp8_prepare,
            minimax_m31_sparse,
        )

        minimax_m31_sparse.register_hooks()
        minimax_m31_decode_graph.register_hooks()
        minimax_m31_draft_swa.register_hooks()
        minimax_m31_fp8_prepare.register_hooks()

    if os.getenv("HCU_PLUGIN_MINIMAX_M31_HISPARSE") == "1":
        from hcu_sglang.models import minimax_m31_hisparse

        minimax_m31_hisparse.register_hooks()

    if os.getenv("HCU_PREFILL_CP", "0") == "1":
        from hcu_sglang.eplb import placement
        from hcu_sglang.layers.attention import minimax_m31_cp

        minimax_m31_cp.register_hooks()
        placement.register_hooks()

    if os.getenv("HCU_PLUGIN_MINIMAX_M31_DEEPEP", "0") == "1":
        from hcu_sglang import deepep_graph
        from hcu_sglang.layers.moe import deepep_current, deepep_fp8, minimax_m31_shared_overlap

        deepep_current.register_hooks()
        deepep_fp8.register_ep_runtime_hooks()
        deepep_graph.register_minimax_hooks()
        minimax_m31_shared_overlap.register_hooks()

    if os.getenv("M31_DIAG_DECODE") == "1":
        from hcu_sglang.models import minimax_m31_decode_diagnostics

        minimax_m31_decode_diagnostics.register_hooks()

    if os.getenv("M31_PROFILE_STAGE") in ("prefill", "decode"):
        from hcu_sglang.models import minimax_m31_profile

        minimax_m31_profile.register_hooks()

    if os.getenv("SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV") == "1":
        from hcu_sglang.models import minimax_m31_router

        minimax_m31_router.register_hooks()
